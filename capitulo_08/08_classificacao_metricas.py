# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 08: Classificação, Matriz de Confusão e F1-Score
# Arquivo: 08_classificacao_metricas.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score

# 1. Gerando um dataset realista desbalanceado (90% Classe 0 - Legítimo, 10% Classe 1 - Fraude)
X, y = make_classification(
    n_samples=1000, 
    n_features=5, 
    weights=[0.90, 0.10], 
    random_state=42
)

# 2. Divisão Estratificada (Mantém a proporção de 10% em ambos os conjuntos)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Treino: {len(y_treino)} amostras (Fraudes: {sum(y_treino)})")
print(f"Teste:  {len(y_teste)} amostras (Fraudes: {sum(y_teste)})")

# 3. Treinando uma Árvore de Decisão com Profundidade Controlada (Evitando Overfitting)
arvore = DecisionTreeClassifier(max_depth=4, random_state=42)
arvore.fit(X_treino, y_treino)

# 4. Predições e Matriz de Confusão
predicoes = arvore.predict(X_teste)
matriz = confusion_matrix(y_teste, predicoes)

tn, fp, fn, tp = matriz.ravel()

print("\n=== MATRIZ DE CONFUSÃO ===")
print(f"Verdadeiros Negativos (TN - Acertos Legítimos): {tn}")
print(f"Falsos Positivos (FP - Alarmes Falsos):         {fp}")
print(f"Falsos Negativos (FN - Fraudes que Passaram):   {fn} (CRÍTICO!)")
print(f"Verdadeiros Positivos (TP - Fraudes Pegas):     {tp}")

# 5. Relatório Detalhado de Métricas de Classificação
print("\n=== RELATÓRIO DE DESEMPENHO (CLASSIFICATION REPORT) ===")
print(classification_report(y_teste, predicoes, target_names=["Legítimo (0)", "Fraude (1)"]))

print(f"Recall da Classe Crítica (Fraude): {recall_score(y_teste, predicoes):.2%}")
print(f"F1-Score Balanceado da Fraude:     {f1_score(y_teste, predicoes):.2%}")
