# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 09: Clustering com K-Means, DBSCAN e Silhouette
# Arquivo: 09_clustering_kmeans_dbscan.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# 1. Gerando dados de clientes sintéticos (Renda vs. Pontuação de Gasto)
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.70, random_state=42)

# 2. Pré-processamento obrigatório: Padronização (Média 0, Variância 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Testando o Método do Cotovelo e Calculando a Silhouette para K de 2 a 6
wcss = []
scores_silhueta = []
faixa_k = range(2, 7)

for k in faixa_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    score = silhouette_score(X_scaled, labels)
    scores_silhueta.append(score)
    print(f"K = {k} | Inércia (WCSS): {km.inertia_:.1f} | Silhouette Score: {score:.4f}")

# 4. Treinando o K-Means com o melhor K (K = 4)
melhor_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_kmeans = melhor_kmeans.fit_predict(X_scaled)

# 5. Aplicando o DBSCAN para detecção de densidade e identificação de anomalias
dbscan = DBSCAN(eps=0.35, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

n_outliers = np.sum(clusters_dbscan == -1)
print(f"\nDBSCAN: Identificou {len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)} clusters e {n_outliers} Outliers (-1)")

# 6. Visualização Gráfica dos Resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot K-Means
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_kmeans, cmap="viridis", alpha=0.7, s=40)
ax1.scatter(melhor_kmeans.cluster_centers_[:, 0], melhor_kmeans.cluster_centers_[:, 1], c="red", marker="X", s=150, label="Centroides")
ax1.set_title(f"Segmentação K-Means (K=4 | Silhouette: {scores_silhueta[2]:.2f})", fontweight="bold")
ax1.legend()

# Plot DBSCAN
scatter = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_dbscan, cmap="plasma", alpha=0.7, s=40)
ax2.set_title(f"Segmentação DBSCAN (Densidade + {n_outliers} Outliers)", fontweight="bold")

plt.tight_layout()
plt.show()
