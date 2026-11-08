# ======================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 09: Agrupamento Inteligente — K-Means, DBSCAN e Silhouette
# Arquivo: 09_clustering_kmeans_dbscan.py
# ======================================================================
# Repositório Oficial de Códigos do Livro
# Execute localmente no VS Code ou copie para o Google Colab.
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# 1. Gerando dados sintéticos de clientes (Renda vs. Pontuação)
X, _ = make_blobs(
    n_samples=500, centers=4, cluster_std=0.70, random_state=42
)

# 2. Padronização obrigatória (Média 0 e Variância 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Método do Cotovelo e Silhouette para K de 2 a 6
wcss = []
scores_silhueta = []
faixa_k = range(2, 7)

for k in faixa_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    score = silhouette_score(X_scaled, labels)
    scores_silhueta.append(score)
    print(
        f"K = {k} | WCSS: {km.inertia_:.1f} | "
        f"Silhouette: {score:.4f}"
    )

# 4. Treinando o K-Means com o melhor K (K = 4)
melhor_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_kmeans = melhor_kmeans.fit_predict(X_scaled)

# 5. Aplicando DBSCAN por densidade e isolando anomalias
dbscan = DBSCAN(eps=0.35, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

n_outliers = int(np.sum(clusters_dbscan == -1))
n_clusters_db = len(set(clusters_dbscan)) - (
    1 if -1 in clusters_dbscan else 0
)
print(
    f"\nDBSCAN: {n_clusters_db} clusters e "
    f"{n_outliers} Outliers (-1)"
)

# 6. Visualização Gráfica dos Resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: K-Means
ax1.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters_kmeans,
    cmap="viridis",
    alpha=0.7,
    s=40,
)
centros = melhor_kmeans.cluster_centers_
ax1.scatter(
    centros[:, 0],
    centros[:, 1],
    c="red",
    marker="X",
    s=150,
    label="Centroides",
)
sil_k4 = scores_silhueta[2]
ax1.set_title(
    f"K-Means (K=4 | Silhouette: {sil_k4:.2f})",
    fontweight="bold",
)
ax1.legend()

# Gráfico 2: DBSCAN
ax2.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters_dbscan,
    cmap="plasma",
    alpha=0.7,
    s=40,
)
ax2.set_title(
    f"DBSCAN (Densidade + {n_outliers} Outliers)",
    fontweight="bold",
)

plt.tight_layout()
# plt.show()
