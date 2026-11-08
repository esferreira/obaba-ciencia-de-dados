# ====================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 04: Contando Histórias com Gráficos — EDA
# Arquivo: 04_analise_exploratoria_eda.py
# ====================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Configurando o estilo visual e gerando dados sintéticos de loja
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

np.random.seed(42)
n_amostras = 500
categorias = ["Eletrônicos", "Moda", "Alimentos", "Livros"]

dados_loja = pd.DataFrame(
    {
        "idade": np.random.normal(35, 10, n_amostras).astype(int),
        "gasto_mensal": (
            np.random.exponential(scale=200, size=n_amostras) + 50
        ),
        "categoria_favorita": np.random.choice(
            categorias, size=n_amostras
        ),
        "tempo_no_app_min": np.random.normal(25, 8, n_amostras),
    }
)

dados_loja["gasto_mensal"] += dados_loja["tempo_no_app_min"] * 8.5

# 2. Histograma e Curva KDE: Distribuição do Gasto Mensal
plt.figure(figsize=(8, 4))
sns.histplot(
    data=dados_loja, x="gasto_mensal", kde=True, color="#2b5c8f"
)
plt.title(
    "Distribuição do Gasto Mensal dos Clientes (em reais)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Gasto Mensal (em reais)")
plt.ylabel("Contagem de Clientes")
plt.tight_layout()
plt.show()

# 3. Gráfico de Dispersão: Tempo de Uso do App vs. Gasto Mensal
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=dados_loja,
    x="tempo_no_app_min",
    y="gasto_mensal",
    hue="categoria_favorita",
    alpha=0.7,
    s=60,
)
plt.title(
    "Tempo de Uso do App vs. Gasto Mensal por Categoria",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Tempo Médio Diário no App (minutos)")
plt.ylabel("Gasto Mensal (em reais)")
plt.legend(
    title="Categoria Preferida",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
plt.tight_layout()
plt.show()

# 4. Boxplot: Dispersão e Detecção de Outliers por Categoria
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=dados_loja,
    x="categoria_favorita",
    y="gasto_mensal",
    hue="categoria_favorita",
    palette="Set2",
    legend=False,
)
plt.title(
    "Dispersão e Detecção de Outliers de Gasto por Categoria",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Categoria de Produto")
plt.ylabel("Gasto Mensal (em reais)")
plt.tight_layout()
plt.show()

# 5. Mapa de Calor (Heatmap): Matriz de Correlação Linear de Pearson
colunas_num = ["idade", "gasto_mensal", "tempo_no_app_min"]
correlacao = dados_loja[colunas_num].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(
    correlacao,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    linewidths=1,
)
plt.title(
    "Mapa de Calor da Correlação Linear",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
