# Capítulo 15 — AutoML com PyCaret e Alternativa em Scikit-Learn
# Arquivo: 15_automl_pycaret.py

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# 1. Gerando base de inadimplência de cartão de crédito
X, y = make_classification(
    n_samples=600,
    n_features=8,
    n_informative=5,
    weights=[0.75, 0.25],
    random_state=42
)
colunas = [f"atributo_{i}" for i in range(1, 9)]
dados_cartao = pd.DataFrame(X, columns=colunas)
dados_cartao["default"] = y  # 1 = Inadimplente, 0 = Pagador

# 2. Tentando executar o pipeline oficial com PyCaret
try:
    from pycaret.classification import (
        setup,
        compare_models,
        tune_model,
        predict_model
    )

    experimento = setup(
        data=dados_cartao,
        target="default",
        train_size=0.80,
        session_id=42,
        normalize=True,
        verbose=False
    )
    print("=== LEADERBOARD DE AUTOML COM PYCARET ===")
    modelo_campeao = compare_models(sort="F1")
    modelo_otimizado = tune_model(modelo_campeao)
    resultados_teste = predict_model(modelo_otimizado)

except ImportError:
    print(
        "[AVISO] PyCaret não detectado neste ambiente Python "
        "(requer Python 3.10/3.11 ou requirements-automl.txt)."
    )
    print(
        "Executando mini-leaderboard de AutoML equivalente "
        "com Scikit-Learn (validação cruzada 5-fold):\n"
    )

    candidatos = {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500)
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metricas = ["accuracy", "roc_auc", "f1"]
    ranking = []

    for nome, modelo in candidatos.items():
        scores = cross_validate(modelo, X, y, cv=cv, scoring=metricas)
        ranking.append({
            "Modelo": nome,
            "Acuracia": scores["test_accuracy"].mean(),
            "ROC_AUC": scores["test_roc_auc"].mean(),
            "F1_Score": scores["test_f1"].mean()
        })

    tabela_automl = (
        pd.DataFrame(ranking)
        .sort_values(by="F1_Score", ascending=False)
        .reset_index(drop=True)
    )
    print("=== MINI-LEADERBOARD DE AUTOML (SCIKIT-LEARN) ===")
    print(tabela_automl.to_string(index=False, float_format="%.4f"))
    campeao = tabela_automl.loc[0, "Modelo"]
    melhor_f1 = tabela_automl.loc[0, "F1_Score"]
    print(f"\nModelo Campeão: {campeao} (F1-Score: {melhor_f1:.4f})")
