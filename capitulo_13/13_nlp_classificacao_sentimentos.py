# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 13: Processamento de Linguagem Natural e Sentimentos
# Arquivo: 13_nlp_classificacao_sentimentos.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Base de dados de avaliações de clientes de um aplicativo
dados_avaliacoes = pd.DataFrame({
    "comentario": [
        "Aplicativo maravilhoso, super rápido e fácil de usar adorei!",
        "Excelente atendimento, resolveram meu problema em minutos parabéns.",
        "Experiência incrível, recomendo para todos os meus amigos.",
        "Péssimo serviço, travou na hora do pagamento e perdi meu dinheiro.",
        "Horrível, não funciona nada, trava toda hora aplicativo inútil.",
        "Atendimento muito ruim e demorado, ninguém responde no suporte.",
        "Muito bom, interface bonita e recursos muito úteis.",
        "Pior experiência que já tive, aplicativo lento e cheio de erros."
    ],
    "sentimento": [1, 1, 1, 0, 0, 0, 1, 0]  # 1 = Positivo, 0 = Negativo
})

# Lista de stopwords comuns em português para descartar
stopwords_pt = ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele", "das", "à", "seu", "sua", "ou", "quando", "muito", "nos", "já", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "depois", "sem", "mesmo", "aos", "seus", "quem", "me", "esse", "eles", "você", "essa", "num", "nem", "suas", "meu", "minha", "toda"]

# 2. Vetorização Inteligente com TF-IDF (Tokenização + Limpeza + Ponderação em 1 passo)
vetorizador = TfidfVectorizer(
    lowercase=True, 
    stop_words=stopwords_pt,
    ngram_range=(1, 1)  # Palavras isoladas
)

X_tfidf = vetorizador.fit_transform(dados_avaliacoes["comentario"])
y = dados_avaliacoes["sentimento"]

print(f"Dimensão da Matriz TF-IDF: {X_tfidf.shape[0]} comentários x {X_tfidf.shape[1]} palavras únicas no vocabulário")

# 3. Treinando o Modelo Classificador (Regressão Logística com Função Sigmoide)
modelo_nlp = LogisticRegression()
modelo_nlp.fit(X_tfidf, y)

# 4. Testando o Modelo em Comentários Inéditos do Mundo Real
novos_comentarios = [
    "Adorei tudo, suporte rápido e produto incrível!",
    "Serviço horrível, tudo péssimo e travando sempre."
]

novos_vetores = vetorizador.transform(novos_comentarios)
previsoes = modelo_nlp.predict(novos_vetores)
probabilidades = modelo_nlp.predict_proba(novos_vetores)

print("\n=== CLASSIFICAÇÃO DE SENTIMENTOS EM COMENTÁRIOS NOVOS ===")
for texto, pred, prob in zip(novos_comentarios, previsoes, probabilidades):
    rotulo = "🟢 POSITIVO" if pred == 1 else "🔴 NEGATIVO"
    certeza = prob[pred] * 100
    print(f"\nComentário: \"{texto}\"")
    print(f"Decisão da IA: {rotulo} (Certeza: {certeza:.1f}%)")
