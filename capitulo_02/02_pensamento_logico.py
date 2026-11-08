# ====================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 02: Pensamento Lógico com Python
# Arquivo: 02_pensamento_logico.py
# ====================================================================

# 1. Variáveis (Caixas Etiquetadas na Memória) e Tipos Primitivos
nome_cliente = "Mariana Souza"
idade_anos = 34
renda_mensal = 6500.50
cliente_ativo = True

print(f"Cliente: {nome_cliente} | Idade: {idade_anos} anos")
print(f"Renda: {renda_mensal:.2f} reais | Ativo: {cliente_ativo}")

# 2. Listas: Sequência Ordenada e Mutável de Faturamento Diário
faturamento_diario = [1250.0, 1800.50, 950.0, 2100.0, 1600.0]
faturamento_diario.append(2300.0)

qtd_dias = len(faturamento_diario)
media_diaria = sum(faturamento_diario) / qtd_dias

print("Quantidade de dias registrados:", qtd_dias)
print("Maior faturamento:", max(faturamento_diario))
print(f"Média diária: {media_diaria:.2f} reais")

# 3. Dicionários: Ficha Cadastral no Formato Chave e Valor
cliente = {
    "id": 1042,
    "nome": nome_cliente,
    "score_credito": 780,
    "renda_mensal": renda_mensal,
    "possui_cartao": True,
    "cidades_frequentes": ["São Paulo", "Campinas"],
}

cliente["score_credito"] += 15
print(
    f"Ficha: {cliente['nome']} | Novo Score: {cliente['score_credito']}"
)

# 4. Conjuntos (set): Eliminando Duplicatas Instantaneamente
categorias_brutas = [
    "Eletrônicos",
    "Moda",
    "Moda",
    "Alimentos",
    "Eletrônicos",
]
categorias_unicas = sorted(set(categorias_brutas))
print("Categorias Únicas:", categorias_unicas)

# 5. Controle de Fluxo: Condicionais (if, elif, else)
score = cliente["score_credito"]
renda = cliente["renda_mensal"]

if score >= 700 and renda >= 4000:
    status_credito = "Aprovado com Limite Alto"
elif score >= 500:
    status_credito = "Aprovado com Limite Básico"
else:
    status_credito = "Reprovado - Risco Elevado"

print(f"Decisão do Sistema: {status_credito}")

# 6. Repetição e List Comprehension: Reajuste de Preços em Lote
precos_antigos = [100.0, 250.0, 400.0, 80.0]
precos_novos = [round(p * 1.10, 2) for p in precos_antigos]
print("Preços Reajustados (+10%):", precos_novos)


# 7. Funções Reutilizáveis com Tipagem Explícita e Docstring
def calcular_imc(peso_kg: float, altura_m: float) -> dict:
    """Calcula o IMC e retorna o valor e a classificação clínica."""
    imc = peso_kg / (altura_m**2)

    if imc < 18.5:
        classificacao = "Abaixo do peso"
    elif imc < 25.0:
        classificacao = "Peso normal"
    elif imc < 30.0:
        classificacao = "Sobrepeso"
    else:
        classificacao = "Obesidade"

    return {"imc": round(imc, 2), "classificacao": classificacao}


resultado = calcular_imc(peso_kg=78.5, altura_m=1.75)
print(f"IMC: {resultado['imc']} ({resultado['classificacao']})")
