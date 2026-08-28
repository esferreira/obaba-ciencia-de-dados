# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 10: Redução de Dimensionalidade com PCA
# Arquivo: 10_reducao_dimensionalidade_pca.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Carregando dados de alta dimensão (30 variáveis biomédicas)
dados = load_breast_cancer()
X = dados.data
y = dados.target
nomes_features = dados.feature_names

print(f"Dimensões Originais: {X.shape[0]} amostras x {X.shape[1]} colunas (Variáveis)")

# 2. Padronização Obrigatória (Média = 0, Desvio Padrão = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aplicando PCA para reduzir de 30 para 2 Componentes Principais
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Avaliando a Variância Retida pelos Componentes
var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
var_total = var_pc1 + var_pc2

print("\n=== PODER DE COMPRESSÃO DO PCA ===")
print(f"Componente Principal 1 (PC1): Retém {var_pc1:.2f}% da variância total")
print(f"Componente Principal 2 (PC2): Retém {var_pc2:.2f}% da variância total")
print(f"Total Retido com apenas 2 Eixos: {var_total:.2f}% de toda a informação original!")

# 5. Criando um DataFrame consolidado com a projeção 2D
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Diagnostico"] = ["Benigno" if rotulo == 1 else "Maligno" for rotulo in y]

# 6. Plotando a Visualização 2D Reveladora dos Dados Originais de 30 Dimensões
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_pca, 
    x="PC1", 
    y="PC2", 
    hue="Diagnostico", 
    palette={"Benigno": "#2b5c8f", "Maligno": "#d9534f"},
    alpha=0.8,
    s=70
)
plt.title(f"Projeção PCA 2D a partir de 30 Variáveis (Retém {var_total:.1f}% da Variância)", fontsize=13, fontweight="bold")
plt.xlabel(f"Componente Principal 1 ({var_pc1:.1f}% da Informação)")
plt.ylabel(f"Componente Principal 2 ({var_pc2:.1f}% da Informação)")
plt.legend(title="Diagnóstico Real")
plt.tight_layout()
plt.show()
