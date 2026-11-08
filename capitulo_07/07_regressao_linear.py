# ====================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 07: Regressão Linear e Regularização
# Arquivo: codigo_livro/capitulo_07/07_regressao_linear.py
# ====================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# 1. Carregando dados (com contingência offline automática)
try:
    dados = fetch_california_housing(as_frame=True)
    X = dados.data
    y = dados.target * 100000
    nome_base = "California Housing (em USD)"
except Exception:
    dados = load_diabetes(as_frame=True)
    X = dados.data
    y = dados.target * 1000
    nome_base = "Diabetes Offline Fallback (escala ajustada)"

print(f"Base ativa: {nome_base}")
print(f"Registros: {X.shape[0]} | Variáveis: {X.shape[1]}")

# 2. Divisão Sagrada: 80% Treino e 20% Teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Padronização ajustada apenas no conjunto de treino
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# 4. Validação Cruzada de 5 Partições (5-Fold) no Treino
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_cv = cross_val_score(
    LinearRegression(),
    X_treino_scaled,
    y_treino,
    cv=kfold,
    scoring="r2",
)
media_cv = scores_cv.mean()
desvio_cv = scores_cv.std()
print(
    f"R² Médio na Validação (5-Fold): {media_cv:.4f} "
    f"(+/- {desvio_cv:.4f})"
)

# 5. Treinando Regressão com Regularização Lasso (L1) e Ridge (L2)
modelo_ridge = Ridge(alpha=10.0, random_state=42)
modelo_ridge.fit(X_treino_scaled, y_treino)

modelo_lasso = Lasso(alpha=500.0, random_state=42)
modelo_lasso.fit(X_treino_scaled, y_treino)

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
print(f"MAE  (Erro Médio Direto):        USD {mae:,.2f}")
print(f"MSE  (Erro Quadrático Médio):    USD {mse:,.2f}")
print(f"RMSE (Penalização Grandes Erros): USD {rmse:,.2f}")
print(f"R²   (Variância Explicada):       {r2 * 100:.2f}%")
