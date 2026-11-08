# Códigos do Livro: O B-A-BA da Ciência de Dados (2ª Edição · 2026)

### *Como Entender Dados, Interpretar Modelos e Decifrar a Inteligência Artificial*

Bem-vindo ao repositório oficial de códigos práticos do livro **O B-A-BA da Ciência de Dados**, de **Écio Soares Ferreira** (`https://github.com/esferreira/obaba-ciencia-de-dados`).

Este repositório contém todos os 16 scripts Python organizados por capítulo, sincronizados linha a linha com o livro e prontos para execução no computador local (VS Code / Terminal) ou no **Google Colab**.

---

## Como Usar Este Repositório

### 1. Clonar no Computador Local
```bash
git clone https://github.com/esferreira/obaba-ciencia-de-dados.git
cd obaba-ciencia-de-dados
```

### 2. Criar Ambiente Virtual e Instalar Dependências (Python 3.11 a 3.14)
```bash
# Criar ambiente virtual isolado
python -m venv .venv

# Ativar no Windows
.venv\Scripts\activate

# Ativar no Linux / macOS
source .venv/bin/activate

# Instalar as dependências principais (Capítulos 01 a 16)
pip install -r requirements.txt
```

> **Nota sobre PyTorch (Capítulo 01) e PyCaret (Capítulo 15):**
> - No **Capítulo 01**, a biblioteca `torch` é **opcional** (usada apenas para inspecionar placas de vídeo NVIDIA/CUDA, se presentes). O script roda normalmente sem ela.
> - No **Capítulo 15 (AutoML)**, o pacote `pycaret` suporta Python 3.10 e 3.11 (ou Google Colab) e está separado em `requirements-automl.txt`. Se você estiver em Python 3.12+, o script `15_automl_pycaret.py` detecta o ambiente automaticamente e executa o comparativo de modelos (*Leaderboard*) usando `scikit-learn` puro.

### 3. Executar os Scripts por Capítulo
```bash
python capitulo_01/01_verificar_ambiente.py
python capitulo_02/02_pensamento_logico.py
python capitulo_03/03_domando_tabelas_pandas.py
python capitulo_16/16_a_bussola_do_cientista_de_dados.py
```

---

## Estrutura de Capítulos

| Módulo | Capítulo | Script Principal |
| :--- | :--- | :--- |
| **M1: Laboratório** | Cap 01 | `capitulo_01/01_verificar_ambiente.py` |
| **M1: Laboratório** | Cap 02 | `capitulo_02/02_pensamento_logico.py` |
| **M1: Laboratório** | Cap 03 | `capitulo_03/03_domando_tabelas_pandas.py` |
| **M1: Laboratório** | Cap 04 | `capitulo_04/04_analise_exploratoria_eda.py` |
| **M2: Estatística e Séries** | Cap 05 | `capitulo_05/05_inferencia_bayesiana.py` |
| **M2: Estatística e Séries** | Cap 06 | `capitulo_06/06_series_temporais.py` |
| **M3: Supervisionado** | Cap 07 | `capitulo_07/07_regressao_linear.py` |
| **M3: Supervisionado** | Cap 08 | `capitulo_08/08_classificacao_metricas.py` |
| **M4: Não-Supervisionado** | Cap 09 | `capitulo_09/09_clustering_kmeans_dbscan.py` |
| **M4: Não-Supervisionado** | Cap 10 | `capitulo_10/10_reducao_dimensionalidade_pca.py` |
| **M5: Leis Matemáticas** | Cap 11 | `capitulo_11/11_lei_de_benford_fraudes.py` |
| **M5: Leis Matemáticas** | Cap 12 | `capitulo_12/12_lei_de_zipf_linguagem.py` |
| **M6: Nova Fronteira** | Cap 13 | `capitulo_13/13_nlp_classificacao_sentimentos.py` |
| **M6: Nova Fronteira** | Cap 14 | `capitulo_14/14_slms_groq_api.py` |
| **M6: Nova Fronteira** | Cap 15 | `capitulo_15/15_automl_pycaret.py` |
| **M6: Nova Fronteira** | Cap 16 | `capitulo_16/16_a_bussola_do_cientista_de_dados.py` |
