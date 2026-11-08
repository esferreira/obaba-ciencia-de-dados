# ====================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 08: Classificação e Métricas
# Arquivo: codigo_livro/capitulo_08/08_classificacao_metricas.py
# ====================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

# 1. Gerando base desbalanceada (90% Legítimo = 0, 10% Fraude = 1)
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    weights=[0.90, 0.10],
    random_state=42,
)

# 2. Divisão Estratificada (Mantém 10% de fraudes em treino e teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Treino: {len(y_treino)} amostras (Fraudes: {sum(y_treino)})")
print(f"Teste:  {len(y_teste)} amostras (Fraudes: {sum(y_teste)})")

# 3. Árvore de Decisão com Profundidade Controlada (max_depth=4)
arvore = DecisionTreeClassifier(max_depth=4, random_state=42)
arvore.fit(X_treino, y_treino)

# 4. Predições e Matriz de Confusão
predicoes = arvore.predict(X_teste)
matriz = confusion_matrix(y_teste, predicoes)

tn, fp, fn, tp = matriz.ravel()

print("\n=== MATRIZ DE CONFUSÃO ===")
print(f"Verdadeiros Negativos (TN - Acertos Legítimos): {tn}")
print(f"Falsos Positivos (FP - Alarmes Falsos):         {fp}")
print(f"Falsos Negativos (FN - Fraudes Omitidas):       {fn}")
print(f"Verdadeiros Positivos (TP - Fraudes Pegas):     {tp}")

# 5. Relatório Detalhado de Métricas de Classificação
nomes_classes = ["Legítimo (0)", "Fraude (1)"]
relatorio = classification_report(
    y_teste, predicoes, target_names=nomes_classes
)
print("\n=== RELATÓRIO DE DESEMPENHO (CLASSIFICATION REPORT) ===")
print(relatorio)

rec = recall_score(y_teste, predicoes)
f1 = f1_score(y_teste, predicoes)
print(f"Recall da Classe Crítica (Fraude): {rec:.2%}")
print(f"F1-Score Balanceado da Fraude:     {f1:.2%}")
