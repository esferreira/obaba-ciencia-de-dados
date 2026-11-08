# ====================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 03: A Arte de Domar Tabelas — Pandas e Limpeza de Dados
# Arquivo: 03_domando_tabelas_pandas.py
# ====================================================================

import numpy as np
import pandas as pd

# 1. Criando um DataFrame simulando registros bancários brutos
dados = {
    "cliente_id": [101, 102, 103, 104, 105, 106, 106],
    "idade": [25, 47, np.nan, 32, 58, 29, 29],
    "renda": [3200.0, 7500.0, 4800.0, np.nan, 12000.0, 3900.0, 3900.0],
    "cidade": [
        "São Paulo",
        "Rio de Janeiro",
        "São Paulo",
        "Curitiba",
        "Curitiba",
        "São Paulo",
        "São Paulo",
    ],
    "inadimplente": [0, 0, 1, 0, 0, 1, 1],
}

df = pd.DataFrame(dados)

print("--- Primeiras 5 linhas (df.head()) ---")
print(df.head())

print("\n--- Resumo Técnico dos Tipos de Coluna (df.info()) ---")
df.info()

print("\n--- Estatísticas Descritivas (df.describe()) ---")
print(df.describe())

# 2. Filtragem Inteligente e Fatiamento Seguro com .copy()
clientes_alta_renda = df[df["renda"] > 5000].copy()
filtro_sp = (df["cidade"] == "São Paulo") & (df["inadimplente"] == 1)
sp_inadimplentes = df.loc[filtro_sp, ["cliente_id", "cidade", "renda"]]

# 3. Limpeza: Removendo Duplicatas e Imputando Valores Ausentes (NaN)
df = df.drop_duplicates().copy()
print("\nValores nulos antes do tratamento:\n", df.isnull().sum())

mediana_idade = df["idade"].median()
df["idade"] = df["idade"].fillna(mediana_idade)

mediana_renda = df["renda"].median()
df["renda"] = df["renda"].fillna(mediana_renda)

# 4. Engenharia de Variáveis (Feature Engineering) Vetorizada
df["faixa_etaria"] = np.where(
    df["idade"] < 30, "Jovem (<30)", "Adulto (30+)"
)
df["renda_em_milhares"] = df["renda"] / 1000.0

# 5. Agregação Analítica por Cidade com groupby
resumo_cidade = (
    df.groupby("cidade")
    .agg(
        renda_media=("renda", "mean"),
        total_clientes=("cliente_id", "count"),
        taxa_inadimplencia=("inadimplente", "mean"),
    )
    .reset_index()
)

print("\n--- Resumo Analítico por Cidade ---")
print(resumo_cidade)
