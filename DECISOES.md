# Decisões Metodológicas e Engenharia de Atributos — Marconi100 (M100 ExaData)

Este documento registra as decisões de engenharia, integração de dados, tratamento de valores faltantes e formulação heurística do alvo para o dataset de telemetria física real do supercomputador **Marconi100 (CINECA)**, visando subsidiar a validação empírica dos modelos de aprendizado de máquina desenvolvidos no artigo *"Classificação do Nível de Risco de Desperdício Ambiental em Racks de Datacenters Voltados a Cargas de IA"* (Coelho et al., SACI 2026).

---

## 1. Fonte dos Dados e Proveniência

* **Registro Primário:** Zenodo Record `7588815` (DOI: [10.5281/zenodo.7588815](https://doi.org/10.5281/zenodo.7588815)).
* **Artigo Descritor:** Borghesi et al., *"M100 ExaData: a data collection campaign on the CINECA's Marconi100 Tier-0 supercomputer"*, *Scientific Data* (Nature Publishing Group), 2023. DOI: `10.1038/s41597-023-02174-3`.
* **Repositório de Ferramentas:** ECS-Lab, Universidade de Bolonha / CINECA ([GitLab ecs-lab/exadata](https://gitlab.com/ecs-lab/exadata)).
* **Escopo:** Conjunto de dados contínuo de 10 meses de produção real (março a dezembro de 2020), particionado no formato colunar **Apache Parquet**.

---

## 2. Plugins e Sensores Reais Integrados

Ao contrário de datasets puramente lógicos baseados em contêineres e software (como o Alibaba Cluster Data), o Marconi100 provê **instrumentação física multissensorial completa de ponta a ponta**:

### A. Telemetria de Hardware e Consumo Elétrico (Plugin IPMI)
* `total_power`: Potência total consumida por nó (em Watts, leitura a cada 20s via BMC). Corresponde ao `active_power_w` do artigo do SACI 2026.
* `p0_power`, `p1_power`: Consumo elétrico individual por socket de processador IBM POWER9 (Watts).
* `pX_mem_power`, `pX_io_power`: Consumo elétrico dos subsistemas de memória e barramentos de E/S.
* `ps0_input_power`, `ps1_input_power`: Potência medida na entrada das fontes redundantes de alimentação (PSUs).
* `ambient`: Temperatura de entrada de ar frio no nó computacional (*inlet temperature*, em °C).
* `pX_coreY_temp`: Temperatura dos núcleos dos processadores (°C).
* `gpuX_core_temp`: Temperatura interna dos chips gráficos NVIDIA Volta V100 (°C). Corresponde a `gpu_temperature_c`.
* `gpuX_mem_temp`: Temperatura dos módulos de memória HBM2 das GPUs (°C).
* `dimmX_temp`: Temperatura dos módulos de memória RAM DDR4 (°C).
* `fanX_Y`, `fan_disk_power`: Velocidade de rotação das ventoinhas internas (RPM) e consumo elétrico de ventilação mecânica.

### B. Infraestrutura Predial e Refrigeração (Schneider & Vertiv)
* `Alm_TY141`: Alarme da válvula de três vias de água gelada do circuito de refrigeração líquida (RDHx).
* `Abilita_inverter`: Estado dos inversores de frequência das bombas de circulação de fluido refrigerante.
* `Return_Air_Temp_Set`: *Set-point* dos aparelhos de ar-condicionado de precisão (CRAC/CRAH).

### C. Cargas de Trabalho e Escalonamento (Slurm / Job Table)
* `job_info_marconi100`: Histórico de execução de jobs com nós físicos alocados (`alloc_node`), tempo em fila (`accrue_time`), duração e recursos computacionais requisitados (`tres_req`).

### D. Clima Meteorológico Externo (Plugin Weather)
* `temp` (°C), `humidity` (%), `pressure` (hPa), `dew_point` (°C): Condições termodinâmicas externas no município de Casalecchio di Reno (Bolonha, Itália), determinando a sobrecarga do sistema de resfriamento predial.

---

## 3. Unidade de Observação e Alinhamento Temporal

* **Identidade da Instância:** A unidade atômica de análise é a tupla `(timestamp_janela, rack_id, node_id)`.
* **Topologia Física de Racks:** 49 racks físicos (Racks 0 a 48), cada um contendo exatamente 20 nós computacionais (IDs 0 a 979) com coordenadas espaciais conhecidas $(X, Y, Z)$.
* **Resolução Temporal:** A amostragem original dos sensores de hardware (20 segundos) é alinhada e agregada em janelas temporais contínuas (reamostragem para intervalos de 10 minutos, com suporte à agregação horária para bater com o padrão do SACI 2026).
* **Tratamento de Dados Faltantes:** Taxa observada de dados faltantes de 0,57%, imputada via interpolação linear temporal contínua por nó antes da agregação, preservando a física do aquecimento e arrefecimento.

---

## 4. Engenharia de Atributos para Risco de Desperdício Ambiental

Seguindo os princípios conceituais consolidados em Coelho et al. (2026), alto consumo elétrico isolado não define desperdício: um nó processando uma carga pesada de aprendizado profundo de forma eficiente possui justificativa técnica para alta potência. O desperdício emerge do desacoplamento entre gasto energético e trabalho útil, ou da ineficiência térmica no resfriamento.

Foram formuladas as seguintes variáveis derivadas:

1. **Diferencial Térmico de GPUs ($\Delta T_{\text{gpu}}$):**
   $$\Delta T_{\text{gpu}} = T_{\text{gpu\_core}} - T_{\text{inlet}}$$
   Mede a elevação de temperatura interna do acelerador em relação ao ar frio admitido no servidor.
2. **Diferencial Térmico de Processadores ($\Delta T_{\text{cpu}}$):**
   $$\Delta T_{\text{cpu}} = T_{\text{cpu\_max}} - T_{\text{inlet}}$$
3. **Densidade de Potência por Rack ($P_{\text{rack}}$):**
   $$P_{\text{rack}} = \sum_{i \in \text{rack}} \text{total\_power}_i$$
   Soma da potência de todos os servidores ativos daquele rack (em kW), correspondendo a `rack_power_density_kw`.
4. **Índice de Sobrecarga de Refrigeração:**
   $$\text{esforco\_refrigeracao} = \frac{\Delta T_{\text{gpu}}}{\text{rotacao\_ventoinhas}} \times T_{\text{externa}}$$
5. **Indicador de Potência Ociosa (*Idle Power Waste*):**
   Sinalizador booleano e percentual indicando quando um nó consome potência estática de linha de base elevada (> 250 W) com temperaturas de GPU frias (< 35 °C) e ausência de jobs alocados.
6. **Score Combinado de Desperdício ($S_{\text{desperdicio}}$):**
   Índice contínuo normalizado no intervalo $[0, 1]$ ponderando subutilização elétrica, dissipação excessiva e sobrecarga climática externa:
   $$S = w_1 \cdot I_{\text{ociosidade}} + w_2 \cdot I_{\text{termico}} + w_3 \cdot I_{\text{refrigeracao}}$$
   onde $w_1 = 0,45$, $w_2 = 0,35$ e $w_3 = 0,20$.

---

## 5. Definição do Alvo Heurístico (`classe_risco_desperdicio`)

O atributo alvo multiclasse preserva exatamente os 3 níveis estudados no SACI 2026:
* **Baixo:** Operação nominal com alta utilização proporcional à potência consumida e arrefecimento eficiente ($S < 0,33$).
* **Moderado:** Pequenos desvios térmicos ou flutuações de potência em regime transitório ($0,33 \le S < 0,66$).
* **Alto:** Nós mantidos em alta potência sem execução de carga útil, superaquecimento severo com limitação térmica (*throttling*) ou operação em desacordo com as condições climáticas externas ($S \ge 0,66$).

---

## 6. Prevenção Rigorosa de Vazamento de Dados (*Data Leakage*)

Para garantir que o conjunto final seja cientificamente válido para submissão à IEEE:
1. **Separação de Preditores vs. Alvo:** Variáveis que compõem a fórmula direta do rótulo não entram como preditoras no treinamento dos classificadores de baseline.
2. **Divisão Temporal Pura (*Time-based Split*):** Os dados são particionados cronologicamente:
   * **Treino:** Primeiros 70% da série temporal.
   * **Validação:** Próximos 15%.
   * **Teste:** 15% finais (dados nunca vistos para avaliação de generalização).
3. **Isolamento de Ajustes de Normalização:** Média, desvio padrão e limites de imputação são calculados estritamente na partição de treino e apenas aplicados em validação e teste.
