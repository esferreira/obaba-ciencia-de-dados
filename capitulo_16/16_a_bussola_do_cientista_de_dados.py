# Capítulo 16 — Diagnóstico Automático com a Bússola de Modelos
# Arquivo: 16_a_bussola_do_cientista_de_dados.py

import pandas as pd


def diagnosticar_problema_dados(
    df, coluna_alvo=None, serie_temporal=False
):
    """Aplica os 5 passos da Bússola para recomendar modelos."""
    n_lin, n_col = df.shape

    # Passo 1 e Passo 3: Natureza do alvo e dependência temporal
    if serie_temporal:
        paradigma = "Séries Temporais"
        modelos = "Holt-Winters, Prophet ou SARIMA"
        metrica = "MAE, RMSE e MAPE"
        divisao = "Corte cronológico (shuffle=False)"
    elif coluna_alvo is None or coluna_alvo not in df.columns:
        paradigma = "Aprendizado Não Supervisionado"
        modelos = "K-Means, DBSCAN ou PCA"
        metrica = "Silhouette Score e Davies-Bouldin"
        divisao = "Validação estrutural de agrupamentos"
    else:
        serie_alvo = df[coluna_alvo].dropna()
        col_texto = df.drop(columns=[coluna_alvo]).select_dtypes(
            include=["object", "string"]
        )
        eh_texto_livre = any(
            df[c].str.len().mean() > 40 for c in col_texto.columns
        )
        n_unicos = serie_alvo.nunique()
        eh_numerico = pd.api.types.is_numeric_dtype(serie_alvo)

        if eh_texto_livre:
            paradigma = "NLP e Modelos de Linguagem (Classificação)"
            modelos = "TF-IDF + Regressão Logística ou SLM via Groq"
            metrica = "F1-Score, Recall e Latência de Inferência"
        elif eh_numerico and n_unicos > 10:
            paradigma = "Supervisionado — Regressão"
            modelos = "Regressão Linear (OLS), Lasso/Ridge ou XGBoost"
            metrica = "MAE, RMSE e R²"
        else:
            paradigma = "Supervisionado — Classificação"
            modelos = (
                "Regressão Logística, Random Forest "
                "ou Gradient Boosting"
            )
            metrica = "F1-Score, Recall, Precision e ROC-AUC"
        divisao = "Treino/Teste estratificado (sem vazamento)"

    # Passo 4 e Passo 5: Explicabilidade e escala para produção
    porte = "Leve/Rápido" if n_lin < 10000 else "Alto Volume"

    return {
        "Dimensao": f"{n_lin} linhas x {n_col} colunas ({porte})",
        "Paradigma": paradigma,
        "Modelos_Recomendados": modelos,
        "Metrica_Principal": metrica,
        "Regra_Validacao": divisao
    }


# Testando a Bússola em 3 cenários clássicos de Ciência de Dados
df_imoveis = pd.DataFrame({
    "area_m2": [
        50, 70, 90, 110, 130, 150, 180, 200, 220, 250, 300
    ],
    "quartos": [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "preco_mil": [
        250, 340, 420, 510, 600, 690, 810, 900, 980, 1150, 1400
    ]
})

df_fraude = pd.DataFrame({
    "valor": [120.0, 4500.0, 35.0, 8900.0, 60.0, 15.0],
    "horario_noturno": [0, 1, 0, 1, 0, 0],
    "fraude": [0, 1, 0, 1, 0, 0]
})

cenarios = [
    ("Previsão de Preço de Imóveis", df_imoveis, "preco_mil", False),
    ("Detecção de Fraude em Cartão", df_fraude, "fraude", False),
    ("Segmentação de Clientes sem Rótulo", df_imoveis, None, False)
]

print("=== DIAGNÓSTICO AUTOMÁTICO COM A BÚSSOLA DE MODELOS ===")
for titulo, base, alvo, temporal in cenarios:
    diag = diagnosticar_problema_dados(base, alvo, temporal)
    print(f"\nCenário: {titulo}")
    for chave, valor in diag.items():
        print(f"  - {chave}: {valor}")
