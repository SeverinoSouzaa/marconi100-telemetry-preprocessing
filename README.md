# Marconi100 Telemetry Preprocessing & Feature Engineering

Repositório dedicado ao pipeline de pré-processamento, engenharia de atributos e análise exploratória de dados de telemetria do supercomputador **Marconi100 (CINECA)**, baseado no dataset oficial **M100 ExaData** (Zenodo Record [7588815](https://doi.org/10.5281/zenodo.7588815)).

## 🎯 Objetivo
Estruturar, limpar e enriquecer s\'eries temporais de telemetria de datacenters reais (consumo elétrico, estresse térmico, refrigeração e cargas de trabalho) para aplicação em modelos de aprendizado de máquina voltados à classificação e predição de risco de desperdício ambiental e eficiência energética.

## 📂 Estrutura do Projeto
- `notebooks/`: Notebooks Jupyter/Colab com análises, gráficos e pipeline de preparação dos dados.
- `src/`: Scripts utilitários de limpeza, reamostragem e extração de atributos.
- `data/`: Diretório local para os dados processados e amostras.
