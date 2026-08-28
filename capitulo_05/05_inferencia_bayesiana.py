# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 05: Inferência Bayesiana e Atualização de Probabilidades
# Arquivo: 05_inferencia_bayesiana.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Configuração visual elegante
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 6))

theta = np.linspace(0, 1, 500)

# 1. Prior: Não temos ideia prévia (distribuição uniforme Beta(1, 1))
a_prior, b_prior = 1, 1
plt.plot(theta, beta.pdf(theta, a_prior, b_prior), label="1. Prior Inicial: Beta(1, 1) - Sem dados", linestyle="--", color="gray")

# 2. Primeira Evidência: 10 acessos, 7 conversões
sucessos_1, total_1 = 7, 10
a_pos1 = a_prior + sucessos_1
b_pos1 = b_prior + (total_1 - sucessos_1)
plt.plot(theta, beta.pdf(theta, a_pos1, b_pos1), label=f"2. Após 10 usuários (7 conversões): Beta({a_pos1}, {b_pos1})", color="orange", linewidth=2)

# 3. Evidência Robusta: Mais 90 acessos (totalizando 65 conversões em 100 acessos)
sucessos_2, total_2 = 65, 100
a_pos2 = a_prior + sucessos_2
b_pos2 = b_prior + (total_2 - sucessos_2)
plt.plot(theta, beta.pdf(theta, a_pos2, b_pos2), label=f"3. Após 100 usuários (65 conversões): Beta({a_pos2}, {b_pos2})", color="#2b5c8f", linewidth=2.5)

# Calculando o intervalo de credibilidade de 95% da posterior final
limite_inferior = beta.ppf(0.025, a_pos2, b_pos2)
limite_superior = beta.ppf(0.975, a_pos2, b_pos2)
plt.axvline(limite_inferior, color="red", linestyle=":", label=f"Intervalo Credível 95% ({limite_inferior:.2f} a {limite_superior:.2f})")
plt.axvline(limite_superior, color="red", linestyle=":")

plt.title("Atualização Contínua de Certezas com Inferência Bayesiana", fontsize=14, fontweight="bold")
plt.xlabel("Taxa Real de Conversão (θ)")
plt.ylabel("Densidade de Probabilidade (Nossa Certeza)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
