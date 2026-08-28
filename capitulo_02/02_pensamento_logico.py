# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 02: Fundamentos de Python e Estruturas de Dados
# Arquivo: 02_pensamento_logico.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

# Lista com o faturamento dos últimos 5 dias
faturamento_diario = [1250.0, 1800.50, 950.0, 2100.0, 1600.0]

# Adicionando um novo dia à lista
faturamento_diario.append(2300.0)

print("Quantidade de dias registrados:", len(faturamento_diario))
print("Maior faturamento:", max(faturamento_diario))
print("Média diária: R$", sum(faturamento_diario) / len(faturamento_diario))

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

# Registro cadastral de um cliente
cliente = {
    "id": 1042,
    "nome": "Mariana Souza",
    "score_credito": 780,
    "renda_mensal": 6500.0,
    "possui_cartao": True,
    "cidades_frequentes": ["São Paulo", "Campinas"]
}

# Acessando e atualizando informações
print(f"Cliente: {cliente['nome']} | Score: {cliente['score_credito']}")
cliente["score_credito"] += 15  # Aumentando o score

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

# Removendo duplicatas com set
categorias_brutas = ["Eletrônicos", "Moda", "Moda", "Alimentos", "Eletrônicos"]
categorias_unicas = set(categorias_brutas)
print("Categorias Únicas:", categorias_unicas)  # {'Moda', 'Eletrônicos', 'Alimentos'}

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

score = 720
renda = 4500.0

if score >= 700 and renda >= 4000:
    status_credito = "Aprovado com Limite Alto"
elif score >= 500:
    status_credito = "Aprovado com Limite Básico"
else:
    status_credito = "Reprovado - Risco Elevado"

print(f"Decisão do Sistema: {status_credito}")

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

# Aplicando um reajuste de 10% em uma lista de preços
precos_antigos = [100.0, 250.0, 400.0, 80.0]

# Forma 1: Usando laço for tradicional
precos_novos = []
for preco in precos_antigos:
    precos_novos.append(round(preco * 1.10, 2))

# Forma 2: List Comprehension (Estilo elegante e pythônico)
precos_novos_direto = [round(p * 1.10, 2) for p in precos_antigos]

print("Preços Reajustados:", precos_novos_direto)

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------

def calcular_indice_massa_corporal(peso_kg: float, altura_m: float) -> dict:
    """Calcula o IMC e retorna o valor e a classificação clínica."""
    imc = peso_kg / (altura_m ** 2)
    
    if imc < 18.5:
        classificacao = "Abaixo do peso"
    elif imc < 25.0:
        classificacao = "Peso normal"
    elif imc < 30.0:
        classificacao = "Sobrepeso"
    else:
        classificacao = "Obesidade"
        
    return {"imc": round(imc, 2), "classificacao": classificacao}

# Usando a função
resultado = calcular_indice_massa_corporal(peso_kg=78.5, altura_m=1.75)
print(f"IMC: {resultado['imc']} ({resultado['classificacao']})")
