"""Pipeline oficial de Pré-processamento e Engenharia de Atributos do Marconi100 (ExaData).

Processa dados reais de telemetria física do supercomputador Marconi100 (CINECA, Zenodo 7588815).
Gera base integrada, dicionário de dados, relatório de qualidade, gráficos e particionamento
para validação de modelos de Machine Learning (Gregory Coelho et al., SACI 2026).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DADOS_RAW = ROOT / 'dados' / 'raw' / 'marconi100_telemetry_real.parquet'
RESULTADOS_DIR = ROOT / 'resultados'
GRAFICOS_DIR = RESULTADOS_DIR / 'graficos'

CONFIG = {
    'seed': 42,
    'split_proporcoes': [0.70, 0.15, 0.15],
    'pesos_desperdicio': {'ociosidade': 0.45, 'estresse_termico': 0.35, 'esforco_cooling': 0.20},
    'limiares_classes': [0.33, 0.66],
    'classes': ['Baixo', 'Moderado', 'Alto'],
}


def obter_turno(hora: int) -> str:
    if 0 <= hora < 6:
        return 'Madrugada'
    elif 6 <= hora < 12:
        return 'Manha'
    elif 12 <= hora < 18:
        return 'Tarde'
    else:
        return 'Noite'


def obter_estacao(mes: int) -> str:
    # Hemisfério Norte (Itália / CINECA)
    if mes in [12, 1, 2]:
        return 'Inverno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verao'
    else:
        return 'Outono'


def carregar_dados_brutos(caminho: Path) -> pd.DataFrame:
    LOG.info(f'Carregando dados brutos de telemetria: {caminho}')
    if not caminho.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {caminho}. Execute download.py primeiro.')
    df = pd.read_parquet(caminho)
    LOG.info(f'Dimensões originais: {df.shape[0]} registros x {df.shape[1]} colunas de telemetria')
    return df


def executar_engenharia_atributos(df: pd.DataFrame) -> pd.DataFrame:
    LOG.info('Executando engenharia de atributos térmicos e energéticos...')
    dados = df.copy()

    # 1. Calendário e Temporal
    dados['timestamp'] = pd.to_datetime(dados['timestamp'])
    dados = dados.sort_values('timestamp').reset_index(drop=True)
    dados['hora_do_dia'] = dados['timestamp'].dt.hour
    dados['dia_da_semana'] = dados['timestamp'].dt.dayofweek
    dados['fim_de_semana'] = (dados['dia_da_semana'] >= 5).astype(int)
    dados['turno'] = dados['hora_do_dia'].apply(obter_turno)
    dados['mes'] = dados['timestamp'].dt.month
    dados['estacao'] = dados['mes'].apply(obter_estacao)

    # 2. Grandezas Elétricas Reais
    dados['potencia_total_w'] = dados['total_power_avg']
    dados['potencia_cpu_w'] = dados['p0_power_avg'] + dados['p1_power_avg']
    dados['potencia_mem_io_w'] = dados['p0_mem_power_avg'] + dados['p1_mem_power_avg'] + dados['p0_io_power_avg'] + dados['p1_io_power_avg']
    dados['potencia_gpu_estimada_w'] = (dados['potencia_total_w'] - dados['potencia_cpu_w'] - dados['potencia_mem_io_w']).clip(lower=0)
    dados['razao_cpu_total'] = dados['potencia_cpu_w'] / (dados['potencia_total_w'] + 1e-6)

    # 3. Grandezas Térmicas e Dissipação
    dados['temperatura_inlet_c'] = dados['ambient_avg']
    gpu_temp_cols = [c for c in ['gpu0_core_temp_avg', 'gpu1_core_temp_avg', 'gpu2_core_temp_avg', 'gpu3_core_temp_avg'] if c in dados.columns]
    dados['temperatura_gpu_media_c'] = dados[gpu_temp_cols].mean(axis=1)
    dados['temperatura_gpu_max_c'] = dados[gpu_temp_cols].max(axis=1)

    # Ventoinhas (RPM)
    fan_cols = [c for c in dados.columns if c.startswith('fan') and c.endswith('_avg') and 'power' not in c]
    dados['rotacao_ventoinhas_rpm'] = dados[fan_cols].mean(axis=1)

    # 4. Diferenciais Térmicos (Engenharia de Atributos)
    dados['delta_t_gpu_inlet'] = dados['temperatura_gpu_media_c'] - dados['temperatura_inlet_c']
    dados['esforco_refrigeracao_rpm_por_c'] = dados['rotacao_ventoinhas_rpm'] / (dados['temperatura_gpu_media_c'] + 1e-6)

    # 5. Indicadores Heurísticos de Desperdício (Coelho et al., SACI 2026)
    # Potência ociosa: servidor em potência basal alta (> 310 W) com delta_t baixo (< 10 °C)
    p_min = dados['potencia_total_w'].quantile(0.05)
    p_max = dados['potencia_total_w'].quantile(0.95)
    norm_potencia = (dados['potencia_total_w'] - p_min) / (p_max - p_min + 1e-6)
    norm_potencia = norm_potencia.clip(0, 1)

    dt_min = dados['delta_t_gpu_inlet'].quantile(0.05)
    dt_max = dados['delta_t_gpu_inlet'].quantile(0.95)
    norm_delta_t = (dados['delta_t_gpu_inlet'] - dt_min) / (dt_max - dt_min + 1e-6)
    norm_delta_t = norm_delta_t.clip(0, 1)

    norm_refrig = (dados['rotacao_ventoinhas_rpm'] - dados['rotacao_ventoinhas_rpm'].min()) / (
        dados['rotacao_ventoinhas_rpm'].max() - dados['rotacao_ventoinhas_rpm'].min() + 1e-6
    )

    # Desperdício = Alta potência com baixa conversão útil (incompatibilidade térmica e elétrica)
    subutilizacao_termica = (norm_potencia - norm_delta_t).clip(lower=0)
    score_bruto = (
        CONFIG['pesos_desperdicio']['ociosidade'] * subutilizacao_termica
        + CONFIG['pesos_desperdicio']['estresse_termico'] * norm_delta_t
        + CONFIG['pesos_desperdicio']['esforco_cooling'] * norm_refrig
    )
    dados['score_desperdicio'] = score_bruto.clip(0, 1)

    # 6. Rótulo Multiclasse (Baixo, Moderado, Alto)
    l1, l2 = CONFIG['limiares_classes']
    condicoes = [
        dados['score_desperdicio'] < l1,
        (dados['score_desperdicio'] >= l1) & (dados['score_desperdicio'] < l2),
        dados['score_desperdicio'] >= l2,
    ]
    dados['classe_risco_desperdicio'] = np.select(condicoes, CONFIG['classes'], default='Moderado')
    LOG.info(f'Distribuição de classes geradas: \n{dados["classe_risco_desperdicio"].value_counts(normalize=True).round(3)}')

    return dados


def dividir_dados_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    LOG.info('Executando split temporal puro (70% treino, 15% validação, 15% teste)...')
    n = len(df)
    n_treino = int(n * CONFIG['split_proporcoes'][0])
    n_val = int(n * (CONFIG['split_proporcoes'][0] + CONFIG['split_proporcoes'][1]))

    treino = df.iloc[:n_treino].copy()
    val = df.iloc[n_treino:n_val].copy()
    teste = df.iloc[n_val:].copy()

    treino['conjunto'] = 'treino'
    val['conjunto'] = 'validacao'
    teste['conjunto'] = 'teste'

    return treino, val, teste


def treinar_modelo_baseline(treino: pd.DataFrame, teste: pd.DataFrame, preditores_num: list[str], preditores_cat: list[str]):
    LOG.info('Treinando classificador baseline Random Forest...')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), preditores_num),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), preditores_cat),
        ]
    )

    X_train = treino[preditores_num + preditores_cat]
    y_train = treino['classe_risco_desperdicio']

    X_test = teste[preditores_num + preditores_cat]
    y_test = teste['classe_risco_desperdicio']

    model = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=CONFIG['seed'], n_jobs=-1))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc_bal = balanced_accuracy_score(y_test, y_pred)
    relat = classification_report(y_test, y_pred, output_dict=True)
    LOG.info(f'Acurácia balanceada no teste: {acc_bal:.4f}')

    return model, acc_bal, relat, y_test, y_pred


def gerar_graficos(df: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray):
    LOG.info('Gerando gráficos analíticos em alta resolução...')
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Distribuição de Classes
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x='classe_risco_desperdicio', order=CONFIG['classes'], palette='viridis')
    plt.title('Distribuição da Classe de Risco de Desperdício (Marconi100)')
    plt.xlabel('Nível de Risco')
    plt.ylabel('Contagem de Registros')
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / 'distribuicao_classes.png', dpi=200)
    plt.close()

    # 2. Potência vs Temperatura por Classe
    plt.figure(figsize=(8, 5))
    sample = df.sample(min(3000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample,
        x='potencia_total_w',
        y='temperatura_gpu_media_c',
        hue='classe_risco_desperdicio',
        palette='viridis',
        alpha=0.6
    )
    plt.title('Relação Potência Ativa (W) vs Temperatura Média de GPU (°C)')
    plt.xlabel('Potência Total do Nó (Watts)')
    plt.ylabel('Temperatura Média de GPU (°C)')
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / 'potencia_vs_temperatura.png', dpi=200)
    plt.close()

    # 3. Boxplot Delta T por Classe
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x='classe_risco_desperdicio', y='delta_t_gpu_inlet', order=CONFIG['classes'], palette='Set2')
    plt.title('Diferencial Térmico GPU - Inlet (ΔT) por Nível de Risco')
    plt.xlabel('Nível de Risco')
    plt.ylabel('ΔT (°C)')
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / 'delta_t_gpu_por_classe.png', dpi=200)
    plt.close()

    # 4. Matriz de Confusão no Teste
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred, labels=CONFIG['classes'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CONFIG['classes'], yticklabels=CONFIG['classes'])
    plt.title('Matriz de Confusão do Modelo Baseline no Teste')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Real')
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / 'matriz_confusao_baseline.png', dpi=200)
    plt.close()


def salvar_resultados(
    base_completa: pd.DataFrame,
    treino: pd.DataFrame,
    val: pd.DataFrame,
    teste: pd.DataFrame,
    modelo: Any,
    acc_bal: float,
    relat: dict,
    preditores_num: list[str],
    preditores_cat: list[str]
):
    LOG.info(f'Salvando artefatos processados em: {RESULTADOS_DIR}')
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Processado Final para Gregory
    base_export = base_completa.copy()
    base_export.to_parquet(RESULTADOS_DIR / 'base_processada.parquet', index=False)
    # Versão CSV (amostrada ou completa com colunas chave para compatibilidade)
    colunas_finais = (
        ['timestamp', 'conjunto', 'classe_risco_desperdicio', 'score_desperdicio']
        + preditores_num
        + preditores_cat
    )
    base_export[colunas_finais].to_csv(RESULTADOS_DIR / 'base_processada.csv', index=False)
    LOG.info(f'Arquivo base_processada.csv gerado com {len(base_export)} linhas!')

    # Partições
    treino.to_parquet(RESULTADOS_DIR / 'treino.parquet', index=False)
    val.to_parquet(RESULTADOS_DIR / 'validacao.parquet', index=False)
    teste.to_parquet(RESULTADOS_DIR / 'teste.parquet', index=False)

    # 2. Modelos
    joblib.dump(modelo, RESULTADOS_DIR / 'modelo_baseline.joblib')

    # 3. Dicionário de Dados
    dicionario = []
    for col in base_export.columns:
        if col in preditores_num or col in preditores_cat:
            papel = 'preditor'
        elif col == 'classe_risco_desperdicio':
            papel = 'alvo_heuristico'
        elif col == 'timestamp':
            papel = 'identificador_temporal'
        elif col == 'conjunto':
            papel = 'particao_treino_teste'
        else:
            papel = 'metrica_auxiliar_telemetria'

        dicionario.append({
            'coluna': col,
            'tipo': str(base_export[col].dtype),
            'papel': papel,
            'nulos_percentual': round(float(base_export[col].isnull().mean() * 100), 2),
            'valores_unicos': int(base_export[col].nunique()),
        })

    pd.DataFrame(dicionario).to_csv(RESULTADOS_DIR / 'dicionario_dados.csv', index=False)

    # 4. Relatório de Qualidade
    qualidade = {
        'total_registros': len(base_completa),
        'total_treino': len(treino),
        'total_validacao': len(val),
        'total_teste': len(teste),
        'acuracia_balanceada_teste': round(acc_bal, 4),
        'distribuicao_classes': base_completa['classe_risco_desperdicio'].value_counts().to_dict(),
        'metricas_classificacao': relat,
    }
    with open(RESULTADOS_DIR / 'relatorio_qualidade.json', 'w', encoding='utf-8') as fp:
        json.dump(qualidade, fp, indent=2)

    with open(RESULTADOS_DIR / 'relatorio_qualidade.md', 'w', encoding='utf-8') as fp:
        fp.write(f"# Relatório de Qualidade do Pré-Processamento (Marconi100)\n\n")
        fp.write(f"- **Total de Instâncias Processadas:** {len(base_completa):,}\n")
        fp.write(f"- **Divisão:** Treino ({len(treino):,}), Validação ({len(val):,}), Teste ({len(teste):,})\n")
        fp.write(f"- **Acurácia Balanceada no Teste (Baseline Random Forest):** {acc_bal * 100:.2f}%\n\n")
        fp.write(f"### Distribuição de Classes de Risco:\n")
        for cls, count in base_completa['classe_risco_desperdicio'].value_counts().items():
            fp.write(f"- **{cls}:** {count:,} ({count / len(base_completa) * 100:.1f}%)\n")


def main():
    df_raw = carregar_dados_brutos(DADOS_RAW)
    df_feat = executar_engenharia_atributos(df_raw)

    treino, val, teste = dividir_dados_temporal(df_feat)
    base_completa = pd.concat([treino, val, teste], axis=0).reset_index(drop=True)

    preditores_num = [
        'potencia_total_w', 'potencia_cpu_w', 'potencia_gpu_estimada_w',
        'temperatura_inlet_c', 'temperatura_gpu_media_c', 'temperatura_gpu_max_c',
        'rotacao_ventoinhas_rpm', 'delta_t_gpu_inlet', 'razao_cpu_total',
        'hora_do_dia', 'dia_da_semana', 'fim_de_semana'
    ]
    preditores_cat = ['turno', 'estacao']

    modelo, acc_bal, relat, y_test, y_pred = treinar_modelo_baseline(
        treino, teste, preditores_num, preditores_cat
    )

    gerar_graficos(base_completa, y_test, y_pred)

    salvar_resultados(
        base_completa, treino, val, teste,
        modelo, acc_bal, relat,
        preditores_num, preditores_cat
    )
    LOG.info('Pipeline concluído com sucesso absoluto!')


if __name__ == '__main__':
    main()
