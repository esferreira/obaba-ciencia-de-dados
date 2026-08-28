# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 03: Manipulação e Limpeza com Pandas
# Arquivo: 03_domando_tabelas_pandas.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------import pandas as pd
import numpy as np

# Criando um DataFrame de exemplo simulando transações bancárias
dados = {
    "cliente_id": [101, 102, 103, 104, 105, 106, 106],  # Note a duplicata 106
    "idade": [25, 47, np.nan, 32, 58, 29, 29],          # Note o valor nulo np.nan
    "renda": [3200.0, 7500.0, 4800.0, np.nan, 12000.0, 3900.0, 3900.0],
    "cidade": ["São Paulo", "Rio de Janeiro", "São Paulo", "Curitiba", "Curitiba", "São Paulo", "São Paulo"],
    "inadimplente": [0, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(dados)

# 1. Visualizando as primeiras linhas
print("--- Primeiras 5 linhas (df.head()) ---")
print(df.head())

# 2. Estrutura e tipos de dados (df.info())
print("\n--- Resumo Técnico dos Tipos de Coluna (df.info()) ---")
df.info()

# 3. Estatísticas descritivas rápidas (df.describe())
print("\n--- Estatísticas Descritivas dos Dados Numéricos ---")
print(df.describe())

# Selecionando apenas clientes com renda acima de R$ 5.000
clientes_alta_renda = df[df["renda"] > 5000]

# Filtrando com múltiplas condições (E = &, OU = |)
# Clientes de São Paulo que já foram inadimplentes
sp_inadimplentes = df[(df["cidade"] == "São Paulo") & (df["inadimplente"] == 1)]

# Selecionando colunas específicas usando .loc (por nome) ou .iloc (por índice numérico)
apenas_contato = df.loc[:, ["cliente_id", "cidade", "renda"]]

# Remove registros idênticos repetidos
df = df.drop_duplicates()

# Verificando quantos nulos existem por coluna
print("Valores nulos por coluna:\n", df.isnull().sum())

# Preenchendo a idade com a mediana das idades
mediana_idade = df["idade"].median()
df["idade"] = df["idade"].fillna(mediana_idade)

# Preenchendo a renda com a média
media_renda = df["renda"].mean()
df["renda"] = df["renda"].fillna(media_renda)

# Calculando a média de renda e taxa de inadimplência por cidade
resumo_cidade = df.groupby("cidade").agg(
    renda_media=("renda", "mean"),
    total_clientes=("cliente_id", "count"),
    taxa_inadimplencia=("inadimplente", "mean")
).reset_index()

print("--- Resumo Analítico por Cidade ---")
print(resumo_cidade)

# Criando uma coluna de Faixa Etária
df["faixa_etaria"] = df["idade"].apply(lambda x: "Jovem (<30)" if x < 30 else "Adulto (30+)")

# Criando um indicador de Renda Per Capita fictícia
df["renda_normalizada"] = df["renda"] / 1000.0
