# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 11: Detecção de Anomalias e Auditoria com Lei de Benford
# Arquivo: 11_lei_de_benford_fraudes.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Função que calcula a distribuição teórica da Lei de Benford
def calcular_benford_teorico():
    digitos = np.arange(1, 10)
    probs = np.log10(1 + 1 / digitos)
    return pd.Series(probs, index=digitos, name="Teorico_Benford")

# 2. Gerando uma base legítima de reembolsos médicos (10.000 consultas)
np.random.seed(42)
n_amostras = 10000

# Valores reais seguem uma distribuição log-normal (várias ordens de magnitude)
valores_legitimos = np.random.lognormal(mean=5.5, sigma=1.2, size=n_amostras)

# 3. Inserindo Fraude Simulada: Fracionando procedimentos caros (> R$ 600) em partes de R$ 190 a R$ 290
valores_com_fraude = []
for val in valores_legitimos:
    if val > 600 and np.random.rand() > 0.4:  # 60% dos procedimentos caros são fraudados
        partes = int(val // 220) + 1
        for _ in range(partes):
            valores_com_fraude.append(val / partes)
    else:
        valores_com_fraude.append(val)

valores_com_fraude = np.array(valores_com_fraude)

# 4. Função para extrair o primeiro dígito de qualquer número
def extrair_primeiro_digito(series_valores):
    primeiros_digitos = [int(str(abs(v)).replace(".", "").lstrip("0")[0]) for v in series_valores if v > 0]
    contagem = pd.Series(primeiros_digitos).value_counts(normalize=True).sort_index()
    return contagem

# 5. Comparação Estatística
df_comparacao = pd.DataFrame({
    "Lei de Benford (Esperado)": calcular_benford_teorico(),
    "Dados Reais Legítimos": extrair_primeiro_digito(valores_legitimos),
    "Dados com Fraude": extrair_primeiro_digito(valores_com_fraude)
})

print("=== DISTRIBUIÇÃO DOS PRIMEIROS DÍGITOS ===")
print((df_comparacao * 100).round(2).astype(str) + "%")

# 6. Plotagem Forense: Revelando a Anomalia no Gráfico
plt.figure(figsize=(10, 6))
x = np.arange(1, 10)
largura = 0.28

plt.bar(x - largura, df_comparacao["Lei de Benford (Esperado)"] * 100, width=largura, label="Benford Teórico", color="#2b5c8f", alpha=0.9)
plt.bar(x, df_comparacao["Dados Reais Legítimos"] * 100, width=largura, label="Reembolsos Legítimos", color="#2ca02c", alpha=0.7)
plt.bar(x + largura, df_comparacao["Dados com Fraude"] * 100, width=largura, label="Base com Fraude Injetada", color="#d9534f", alpha=0.8)

plt.title("Auditoria Forense com a Lei de Benford: Detecção de Fraudes", fontsize=14, fontweight="bold")
plt.xlabel("Primeiro Dígito da Fatura")
plt.ylabel("Frequência de Ocorrência (%)")
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.show()
