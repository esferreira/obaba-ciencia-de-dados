# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 04: Visualização de Dados com Seaborn e Matplotlib
# Arquivo: 04_analise_exploratoria_eda.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configurando o estilo visual moderno
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

# Gerando dados sintéticos simulados de e-commerce
np.random.seed(42)
n_amostras = 500

dados_loja = pd.DataFrame({
    "idade": np.random.normal(35, 10, n_amostras).astype(int),
    "gasto_mensal": np.random.exponential(scale=200, size=n_amostras) + 50,
    "categoria_favorita": np.random.choice(["Eletrônicos", "Moda", "Alimentos", "Livros"], size=n_amostras),
    "tempo_no_app_min": np.random.normal(25, 8, n_amostras)
})

# Adicionando uma correlação positiva entre tempo no app e gasto
dados_loja["gasto_mensal"] += dados_loja["tempo_no_app_min"] * 8.5

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

plt.figure(figsize=(8, 4))
sns.histplot(data=dados_loja, x="gasto_mensal", kde=True, color="#2b5c8f")
plt.title("Distribuição do Gasto Mensal dos Clientes (R$)", fontsize=14, fontweight="bold")
plt.xlabel("Gasto Mensal (R$)")
plt.ylabel("Contagem de Clientes")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=dados_loja, 
    x="tempo_no_app_min", 
    y="gasto_mensal", 
    hue="categoria_favorita", 
    alpha=0.7,
    s=60
)
plt.title("Tempo de Uso do App vs. Gasto Mensal por Categoria", fontsize=14, fontweight="bold")
plt.xlabel("Tempo Médio Diário no App (minutos)")
plt.ylabel("Gasto Mensal (R$)")
plt.legend(title="Categoria Preferida", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(data=dados_loja, x="categoria_favorita", y="gasto_mensal", palette="Set2")
plt.title("Dispersão e Outliers de Gasto por Categoria", fontsize=14, fontweight="bold")
plt.xlabel("Categoria de Produto")
plt.ylabel("Gasto Mensal (R$)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

# Calculando a correlação apenas entre variáveis numéricas
correlacao = dados_loja[["idade", "gasto_mensal", "tempo_no_app_min"]].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, linewidths=1)
plt.title("Mapa de Calor de Correlação Linear", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
