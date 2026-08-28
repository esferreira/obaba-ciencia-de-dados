# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 07: Regressão Linear e Métricas de Erro (MAE, RMSE, R2)
# Arquivo: 07_regressao_linear.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Carregando dados reais de habitação
dados = fetch_california_housing(as_frame=True)
X = dados.data
y = dados.target * 100000  # Convertendo para valor real em dólares ($)

print(f"Total de Registros: {X.shape[0]} | Número de Features: {X.shape[1]}")

# 2. Divisão Sagrada: 80% Treino e 20% Teste (com semente fixa para reprodutibilidade)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Pré-processamento: Padronização (Média = 0, Desvio Padrão = 1)
# O scaler deve ser ajustado APENAS no treino para evitar vazamento de dados!
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# 4. Validação Cruzada de 5 Partições (5-Fold) no conjunto de Treino
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_cv = cross_val_score(LinearRegression(), X_treino_scaled, y_treino, cv=kfold, scoring="r2")
print(f"R² Médio na Validação Cruzada (5-Fold): {scores_cv.mean():.4f} (+/- {scores_cv.std():.4f})")

# 5. Treinando o Modelo com Regularização Lasso (L1) para Seleção de Variáveis
modelo_lasso = Lasso(alpha=500.0, random_state=42)
modelo_lasso.fit(X_treino_scaled, y_treino)

# Observando quais coeficientes o Lasso manteve e quais zerou:
coeficientes = pd.Series(modelo_lasso.coef_, index=X.columns)
print("\n--- COEFICIENTES APRENDIDOS PELO LASSO (L1) ---")
print(coeficientes.round(2))

# 6. Avaliação Final nos Dados de Teste (Dados Nunca Vistos)
predicoes_teste = modelo_lasso.predict(X_teste_scaled)

mae = mean_absolute_error(y_teste, predicoes_teste)
mse = mean_squared_error(y_teste, predicoes_teste)
rmse = np.sqrt(mse)
r2 = r2_score(y_teste, predicoes_teste)

print("\n=== DESEMPENHO DO MODELO NO CONJUNTO DE TESTE ===")
print(f"MAE  (Erro Médio Direto):        $ {mae:,.2f}")
print(f"MSE  (Erro Quadrático Médio):   $ {mse:,.2f}")
print(f"RMSE (Penalização Grandes Erros): $ {rmse:,.2f}")
print(f"R²   (Variância Explicada):       {r2 * 100:.2f}%")
