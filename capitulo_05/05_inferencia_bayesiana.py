# ====================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 05: Inferência Bayesiana para Iniciantes
# Arquivo: codigo_livro/capitulo_05/05_inferencia_bayesiana.py
# ====================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Configuração visual do gráfico
estilo = (
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.style.use(estilo)
fig, ax = plt.subplots(figsize=(10, 6))

theta = np.linspace(0, 1, 500)

# 1. Prior: Sem ideia prévia (distribuição uniforme Beta(1, 1))
a_prior, b_prior = 1, 1
plt.plot(
    theta,
    beta.pdf(theta, a_prior, b_prior),
    label="1. Prior Inicial: Beta(1, 1) - Sem dados",
    linestyle="--",
    color="gray",
)

# 2. Primeira Evidência: 10 acessos e 7 conversões
sucessos_1, total_1 = 7, 10
a_pos1 = a_prior + sucessos_1
b_pos1 = b_prior + (total_1 - sucessos_1)
rotulo_1 = (
    f"2. Após 10 usuários (7 conversões): Beta({a_pos1}, {b_pos1})"
)
plt.plot(
    theta,
    beta.pdf(theta, a_pos1, b_pos1),
    label=rotulo_1,
    color="orange",
    linewidth=2,
)

# 3. Evidência Robusta: 65 conversões acumuladas em 100 acessos
sucessos_2, total_2 = 65, 100
a_pos2 = a_prior + sucessos_2
b_pos2 = b_prior + (total_2 - sucessos_2)
rotulo_2 = (
    f"3. Após 100 usuários (65 conversões): Beta({a_pos2}, {b_pos2})"
)
plt.plot(
    theta,
    beta.pdf(theta, a_pos2, b_pos2),
    label=rotulo_2,
    color="#2b5c8f",
    linewidth=2.5,
)

# Calculando o intervalo de credibilidade de 95% da posterior final
lim_inf = beta.ppf(0.025, a_pos2, b_pos2)
lim_sup = beta.ppf(0.975, a_pos2, b_pos2)
rotulo_ic = (
    f"Intervalo Credível 95% ({lim_inf:.2f} a {lim_sup:.2f})"
)
plt.axvline(lim_inf, color="red", linestyle=":", label=rotulo_ic)
plt.axvline(lim_sup, color="red", linestyle=":")

print(f"Posterior final: Beta({a_pos2}, {b_pos2})")
print(f"Intervalo de Credibilidade 95%: {lim_inf:.3f} a {lim_sup:.3f}")

plt.title(
    "Atualização Contínua de Certezas com Inferência Bayesiana",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Taxa Real de Conversão (θ)")
plt.ylabel("Densidade de Probabilidade (Nossa Certeza)")
plt.legend(loc="upper left")
plt.tight_layout()
if "agg" not in plt.get_backend().lower():
    plt.show()
