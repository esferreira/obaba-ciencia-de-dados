# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 15: AutoML e Comparação Automatizada de Modelos com PyCaret
# Arquivo: 15_automl_pycaret.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

# Para instalar no seu ambiente: pip install pycaret
import pandas as pd
from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models, tune_model, plot_model, predict_model

# 1. Carregando um conjunto de dados real de clientes de cartão de crédito
dados_cartao = get_data("credit")

# 2. Configurando o Ambiente de AutoML (1 linha de código automatiza todo o pré-processamento!)
experimento = setup(
    data=dados_cartao, 
    target="default",          # Coluna alvo: 1 = Inadimplente, 0 = Pagador
    train_size=0.80,           # 80% Treino e 20% Teste
    session_id=42,             # Semente fixa para reprodutibilidade
    normalize=True,            # Aplica padronização automática
    transformation=True,       # Trata assimetrias na distribuição
    verbose=False
)

# 3. O Momento Mágico: Comparando Todos os Modelos do Mercado Simultaneamente!
# O PyCaret treinará Regressão Logística, Random Forest, Extra Trees, LightGBM, SVM, etc.
# e exibirá a tabela comparativa ordenada por F1-Score ou Acurácia
print("=== TREINANDO E COMPARANDO TODOS OS MODELOS DISPONÍVEIS ===")
modelo_campeao = compare_models(sort="F1")

# 4. Ajustando os Hiperparâmetros do Modelo Vencedor
modelo_otimizado = tune_model(modelo_campeao)

# 5. Avaliando o Desempenho no Conjunto de Teste
print("\n=== AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ===")
resultados_teste = predict_model(modelo_otimizado)
