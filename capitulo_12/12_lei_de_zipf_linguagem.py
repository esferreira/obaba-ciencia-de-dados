# ======================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 12: A Frequência da Linguagem — A Lei de Zipf
# Arquivo: 12_lei_de_zipf_linguagem.py
# ======================================================================
# Repositório Oficial de Códigos do Livro
# Execute localmente no VS Code ou copie para o Google Colab.
# ======================================================================

import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Carregamento de corpus textual em português
texto_exemplo = (
    "A ciência de dados é a arte de transformar dados brutos "
    "em decisões inteligentes. Os modelos de inteligência "
    "artificial aprendem com os padrões do passado para "
    "antecipar o futuro. No entanto, o cientista de dados "
    "precisa entender a matemática, a lógica e o contexto de "
    "negócio antes de aplicar qualquer algoritmo nos dados. "
    "Os dados revelam a verdade quando sabemos fazer as "
    "perguntas certas aos dados. A análise exploratória "
    "permite identificar anomalias, distribuições "
    "estatísticas, correlações e estruturas ocultas. Em "
    "processamento de linguagem natural, as palavras que os "
    "seres humanos escrevem e falam seguem uma organização "
    "surpreendente. Escritores, jornalistas, poetas e "
    "cientistas utilizam um vocabulário que parece infinito, "
    "mas que no fundo obedece a regras de frequência estritas. "
) * 30

# 2. Limpeza léxica: minúsculas e extração com regex
padrao = r"\b[a-záéíóúâêîôûãõç]+\b"
palavras = re.findall(padrao, texto_exemplo.lower())
total_palavras = len(palavras)
vocab_unico = len(set(palavras))
print(
    f"Total de Tokens: {total_palavras} | "
    f"Vocabulário Único: {vocab_unico}"
)

# 3. Contagem de frequência e ordenação por ranking
contagem = Counter(palavras)
df_zipf = pd.DataFrame(
    contagem.most_common(), columns=["Palavra", "Frequencia"]
)
df_zipf["Rank"] = np.arange(1, len(df_zipf) + 1)

print("\n--- TOP 5 PALAVRAS MAIS FREQUENTES ---")
print(df_zipf.head())

# 4. Ajuste da Equação da Lei de Zipf: f(r) = C / (r^s)
def modelo_zipf(r, c_const, s_exp):
    return c_const / (r ** s_exp)

chute_inicial = [df_zipf["Frequencia"].iloc[0], 1.0]
popt, _ = curve_fit(
    modelo_zipf,
    df_zipf["Rank"],
    df_zipf["Frequencia"],
    p0=chute_inicial,
)
c_estimado, s_estimado = popt
print(f"\nExpoente de Zipf Ajustado (s): {s_estimado:.3f}")
print(
    "Nota: Amostras curtas geram s < 1.0; em livros "
    "inteiros de grande vocabulário, s converge para ~1.0."
)

# 5. Visualização no Espaço Log-Log
plt.figure(figsize=(10, 6))
plt.scatter(
    df_zipf["Rank"],
    df_zipf["Frequencia"],
    label="Frequência Observada",
    color="#2563EB",
    alpha=0.75,
    s=50,
)
curva_ajustada = modelo_zipf(df_zipf["Rank"], *popt)
plt.plot(
    df_zipf["Rank"],
    curva_ajustada,
    color="#DC2626",
    linestyle="--",
    linewidth=2.5,
    label=f"Ajuste de Zipf (s = {s_estimado:.2f})",
)

plt.xscale("log")
plt.yscale("log")
plt.title(
    "A Lei de Zipf em Ação (Escala Log-Log)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Posição no Ranking (Log)")
plt.ylabel("Frequência da Palavra (Log)")
plt.grid(True, which="both", ls=":", color="#CBD5E1")
plt.legend()
plt.tight_layout()
# plt.show()
