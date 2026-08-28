# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 06: Decomposição e Análise de Séries Temporais
# Arquivo: 06_series_temporais.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Criando uma Série Temporal Simulada (2 anos de dados diários com tendência e sazonalidade semanal)
np.random.seed(42)
datas = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
n_dias = len(datas)

tendencia = np.linspace(50, 120, n_dias)                                # Tendência de alta
sazonalidade_semanal = 15 * np.sin(2 * np.pi * datas.dayofweek / 7)     # Padrão semanal (fins de semana)
ruido = np.random.normal(0, 5, n_dias)                                  # Variação do dia

valores = tendencia + sazonalidade_semanal + ruido
serie = pd.Series(valores, index=datas, name="Demanda_Diaria")

# 2. Decomposição da Série (Tendência, Sazonalidade e Ruído)
decomposicao = seasonal_decompose(serie, model="additive", period=7)

fig = decomposicao.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.show()

# 3. Divisão Cronológica em Treino (2024 até Out/2025) e Teste (Últimos 60 dias)
treino = serie.iloc[:-60]
teste = serie.iloc[-60:]

print(f"Total de Dias no Treino: {len(treino)} dias")
print(f"Total de Dias no Teste:  {len(teste)} dias")

# 4. Treinando o Modelo Holt-Winters (Tendência Aditiva + Sazonalidade Aditiva de 7 dias)
modelo = ExponentialSmoothing(
    treino, 
    trend="add", 
    seasonal="add", 
    seasonal_periods=7
).fit()

# Gerando previsões para os 60 dias do conjunto de teste
previsoes = modelo.forecast(steps=60)

# 5. Avaliação do Modelo no Conjunto de Teste
mae = mean_absolute_error(teste, previsoes)
rmse = np.sqrt(mean_squared_error(teste, previsoes))
mape = np.mean(np.abs((teste - previsoes) / teste)) * 100

print("\n=== AVALIAÇÃO DE DESEMPENHO NO TESTE ===")
print(f"MAE (Erro Médio Absoluto): {mae:.2f} unidades")
print(f"RMSE (Penalização de Grandes Erros): {rmse:.2f} unidades")
print(f"MAPE (Erro Percentual Médio): {mape:.2f}%")

# 6. Gráfico Comparativo: Real vs. Previsto
plt.figure(figsize=(10, 5))
plt.plot(treino.index[-90:], treino.values[-90:], label="Histórico de Treino (Últimos 90d)", color="gray")
plt.plot(teste.index, teste.values, label="Dados Reais de Teste (Gabarito)", color="blue", linewidth=2)
plt.plot(teste.index, previsoes.values, label=f"Previsão Holt-Winters (MAPE: {mape:.1f}%)", color="red", linestyle="--", linewidth=2)
plt.title("Previsão de Séries Temporais: Dados Reais vs. Modelo Preditivo", fontsize=14, fontweight="bold")
plt.xlabel("Data")
plt.ylabel("Demanda")
plt.legend()
plt.tight_layout()
plt.show()
