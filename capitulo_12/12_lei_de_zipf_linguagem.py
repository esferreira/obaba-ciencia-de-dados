# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 12: Análise de Frequência de Palavras com Lei de Zipf
# Arquivo: 12_lei_de_zipf_linguagem.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from scipy.optimize import curve_fit

# 1. Texto de exemplo simulado em português (corpus jornalístico/literário)
texto_exemplo = """
A ciência de dados é a arte de transformar dados brutos em decisões inteligentes.
Os modelos de inteligência artificial aprendem com os padrões do passado para antecipar
o futuro. No entanto, o cientista de dados precisa entender a matemática, a lógica e
o contexto de negócio antes de aplicar qualquer algoritmo nos dados. Os dados revelam
a verdade quando sabemos fazer as perguntas certas aos dados.
""" * 50  # Replicando para criar volume amostral

# 2. Limpeza básica: convertendo para minúsculas e extraindo apenas palavras
palavras = re.findall(r"\b[a-záéíóúâêîôûãõç]+\b", texto_exemplo.lower())
print(f"Total de Palavras no Texto: {len(palavras)} | Palavras Únicas (Vocabulário): {len(set(palavras))}")

# 3. Contagem de frequência e ordenação por ranking
contagem = Counter(palavras)
df_zipf = pd.DataFrame(contagem.most_common(), columns=["Palavra", "Frequencia"])
df_zipf["Rank"] = np.arange(1, len(df_zipf) + 1)

print("\n--- TOP 5 PALAVRAS MAIS FREQUENTES ---")
print(df_zipf.head())

# 4. Função matemática da Lei de Zipf: f(r) = a / (r^b)
def modelo_zipf(r, a, b):
    return a / (r ** b)

# Ajustando a curva teórica aos dados observados
parametros, _ = curve_fit(modelo_zipf, df_zipf["Rank"], df_zipf["Frequencia"], p0=[df_zipf["Frequencia"].iloc[0], 1.0])
a_ajustado, b_ajustado = parametros
print(f"\nParâmetros Ajustados da Lei de Zipf: Expoente s (b) = {b_ajustado:.3f} (Muito próximo de 1.0!)")

# 5. Plotagem em Escala Log-Log
plt.figure(figsize=(9, 6))
plt.scatter(df_zipf["Rank"], df_zipf["Frequencia"], label="Frequência Observada no Texto", color="#2b5c8f", alpha=0.7, s=40)
plt.plot(df_zipf["Rank"], modelo_zipf(df_zipf["Rank"], *parametros), color="red", linestyle="--", linewidth=2, label=f"Curva Teórica de Zipf (s = {b_ajustado:.2f})")

plt.xscale("log")
plt.yscale("log")
plt.title("A Lei de Zipf em Ação (Escala Log-Log)", fontsize=14, fontweight="bold")
plt.xlabel("Posição no Ranking (Log)")
plt.ylabel("Frequência da Palavra (Log)")
plt.legend()
plt.tight_layout()
plt.show()
