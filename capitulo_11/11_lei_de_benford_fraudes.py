# ======================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 11: O Detector de Fraudes Natural — A Lei de Benford
# Arquivo: 11_lei_de_benford_fraudes.py
# ======================================================================
# Repositório Oficial de Códigos do Livro
# Execute localmente no VS Code ou copie para o Google Colab.
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Função da distribuição teórica da Lei de Benford
def calcular_benford_teorico():
    digitos = np.arange(1, 10)
    probs = np.log10(1 + 1 / digitos)
    return pd.Series(probs, index=digitos, name="Teorico_Benford")

# 2. Gerando base legítima de reembolsos médicos (10.000 notas)
np.random.seed(42)
n_amostras = 10000

# Valores reais em distribuição log-normal (várias magnitudes)
valores_legitimos = np.random.lognormal(
    mean=5.5, sigma=1.2, size=n_amostras
)

# 3. Fraude simulada: fracionando notas caras (> 600 reais)
valores_com_fraude = []
for val in valores_legitimos:
    if val > 600 and np.random.rand() > 0.4:
        partes = int(val // 220) + 1
        for _ in range(partes):
            valores_com_fraude.append(val / partes)
    else:
        valores_com_fraude.append(val)

valores_com_fraude = np.array(valores_com_fraude)

# 4. Função para extrair o primeiro dígito significativo
def extrair_primeiro_digito(series_valores):
    digitos = [
        int(str(abs(v)).replace(".", "").lstrip("0")[0])
        for v in series_valores
        if v > 0
    ]
    contagem = (
        pd.Series(digitos)
        .value_counts(normalize=True)
        .sort_index()
    )
    return contagem.reindex(range(1, 10), fill_value=0)

# 5. Comparação Estatística em Percentual (%)
benford_esp = calcular_benford_teorico() * 100
legitimos_pct = extrair_primeiro_digito(valores_legitimos) * 100
fraude_pct = extrair_primeiro_digito(valores_com_fraude) * 100

df_comparacao = pd.DataFrame({
    "Benford (Esperado)": benford_esp,
    "Reais Legítimos": legitimos_pct,
    "Com Fraude": fraude_pct,
})

print("=== DISTRIBUIÇÃO DOS PRIMEIROS DÍGITOS (%) ===")
print(df_comparacao.round(2))
