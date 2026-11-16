# Marconi100 Telemetry — Preprocessing & Feature Engineering

Pipeline de pré-processamento auditável, engenharia de atributos e análise exploratória sobre telemetria física real do supercomputador **Marconi100 (CINECA)**, baseado no dataset oficial **M100 ExaData** ([Zenodo Record 7588815](https://doi.org/10.5281/zenodo.7588815)) e no repositório oficial do [ECS-Lab GitLab](https://gitlab.com/ecs-lab/exadata).

Este trabalho constitui a **validação empírica com telemetria observacional real** continuada a partir da formulação metodológica proposta em:
> *Gregory Gabriel Ozaki Coelho, Ana Paula Guimarães Xavier, Tiago dos Santos Mendonça, Gabriel Batista dos Santos, Calil Lima Pereira, Wamberson Pacheco Araújo*.  
> **"Classificação do Nível de Risco de Desperdício Ambiental em Racks de Datacenters Voltados a Cargas de IA"**.  
> *Simpósio Amazônico de Computação e Inovação (SACI 2026)*.

---

## 🎯 Objetivo

Integrar grandezas elétricas reais (potência em Watts), térmicas (temperatura de entrada *inlet*, temperatura de processadores e GPUs NVIDIA Volta V100), refrigeração mecânica e temporalidade contínua para classificar e prever o **nível de risco de desperdício ambiental** (`classe_risco_desperdicio`: *Baixo*, *Moderado*, *Alto*).

Diferente de traces baseados exclusivamente em logs lógicos de contêineres e tarefas (como Alibaba Cluster Data), o dataset Marconi100 provê **sensores físicos reais de hardware e arrefecimento**, permitindo correlacionar potência ativa e temperatura real de dissipação.

Consulte [DECISOES.md](DECISOES.md) para detalhes completos de fórmulas, justificativas e prevenção de vazamento de dados (*data leakage*).

---

## 📂 Estrutura do Repositório

```text
marconi100-telemetry-preprocessing/
│
├── .gitignore                      # Proteção contra envio acidental de dados pesados
├── requirements.txt                # Dependências: pandas, pyarrow, scikit-learn, matplotlib, seaborn
├── README.md                       # Guia geral e instruções de execução
├── DECISOES.md                     # Decisões metodológicas e fundamentação física
├── download.py                     # Script para download automatizado de pacotes do Zenodo
├── pipeline.py                     # Motor principal de limpeza, feature engineering e exportação
│
├── notebooks/
│   └── analise_e_visualizacao.ipynb # Notebook interativo (Colab / Jupyter) com gráficos
│
└── resultados/                     # PRODUTO FINAL (Pronto para consumo pelo Gregory)
    ├── base_processada.csv         # Base tabular limpa unificada (86.615 registros)
    ├── base_processada.parquet     # Versão colunar compacta de alta velocidade
    ├── treino.parquet              # Partição de Treino (70% temporal)
    ├── validacao.parquet           # Partição de Validação (15% temporal)
    ├── teste.parquet               # Partição de Teste (15% temporal)
    ├── dicionario_dados.csv        # Dicionário com papel, tipo e nulos de cada coluna
    ├── relatorio_qualidade.md      # Relatório com estatísticas descritivas
    ├── modelo_baseline.joblib      # Classificador Random Forest treinado (acurácia > 99%)
    └── graficos/                   # Visualizações em alta resolução (PNG)
        ├── distribuicao_classes.png
        ├── potencia_vs_temperatura.png
        ├── delta_t_gpu_por_classe.png
        └── matriz_confusao_baseline.png
```

---

## 🚀 Como Executar

### 1. Clonar o Repositório e Preparar o Ambiente

```bash
git clone https://github.com/SeverinoSouzaa/marconi100-telemetry-preprocessing.git
cd marconi100-telemetry-preprocessing

# Criar e ativar o ambiente virtual (opcional, recomendado)
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar o Pipeline de Ponta a Ponta

```bash
python pipeline.py
```

O script irá:
1. Carregar a base de telemetria física real do Marconi100;
2. Realizar o alinhamento temporal e a engenharia de atributos térmicos ($\Delta T_{\text{gpu}} = T_{\text{gpu}} - T_{\text{inlet}}$);
3. Gerar a rotulação multiclasse de risco de desperdício (`Baixo`, `Moderado`, `Alto`);
4. Realizar o split cronológico estrito (70% treino, 15% validação, 15% teste);
5. Treinar o modelo baseline Random Forest;
6. Salvar os gráficos em `resultados/graficos/` e exportar `base_processada.csv` e `.parquet`.

---

## ⚡ Como o Gregory Pode Usar os Dados Diretamente

Para carregar os dados pré-processados e rodar testes de modelos no Python em menos de 10 segundos:

```python
import pandas as pd
from sklearn.metrics import classification_report

# 1. Carregar a base processada
df = pd.read_parquet('resultados/base_processada.parquet')

# Ou carregar diretamente as partições já divididas sem risco de vazamento:
treino = pd.read_parquet('resultados/treino.parquet')
teste = pd.read_parquet('resultados/teste.parquet')

preditores = [
    'potencia_total_w', 'potencia_cpu_w', 'potencia_gpu_estimada_w',
    'temperatura_inlet_c', 'temperatura_gpu_media_c', 'temperatura_gpu_max_c',
    'rotacao_ventoinhas_rpm', 'delta_t_gpu_inlet', 'razao_cpu_total',
    'hora_do_dia', 'dia_da_semana', 'fim_de_semana'
]

X_train, y_train = treino[preditores], treino['classe_risco_desperdicio']
X_test, y_test = teste[preditores], teste['classe_risco_desperdicio']

print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
```

---

## 🔬 Referências

1. **Dataset Oficial:** Borghesi, A. et al. (2023). *M100 dataset 1: from 20-03 to 20-12*. Zenodo. DOI: [10.5281/zenodo.7588815](https://doi.org/10.5281/zenodo.7588815).
2. **Artigo Descritor:** Borghesi, A. et al. (2023). *M100 ExaData: a data collection campaign on the CINECA's Marconi100 Tier-0 supercomputer*. *Scientific Data* (Nature), 10:292. DOI: [10.1038/s41597-023-02174-3](https://doi.org/10.1038/s41597-023-02174-3).
3. **Repositório de Código Oficial:** ECS-Lab, CINECA / Universidade de Bolonha. [GitLab ecs-lab/exadata](https://gitlab.com/ecs-lab/exadata).
4. **Artigo Base do Grupo:** Coelho, G. G. O. et al. (2026). *Classificação do Nível de Risco de Desperdício Ambiental em Racks de Datacenters Voltados a Cargas de IA*. *SACI 2026*.
