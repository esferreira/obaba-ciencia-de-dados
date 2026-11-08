# Capítulo 13 — Classificador de Sentimentos com TF-IDF
# Arquivo: 13_nlp_classificacao_sentimentos.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Base de avaliações de clientes de um aplicativo
dados_avaliacoes = pd.DataFrame({
    "comentario": [
        "Aplicativo maravilhoso, super rápido e fácil de usar!",
        "Excelente atendimento, resolveram meu problema rápido.",
        "Experiência incrível, recomendo para todos os amigos.",
        "Péssimo serviço, travou no pagamento e perdi dinheiro.",
        "Horrível, não funciona nada, trava toda hora, inútil.",
        "Atendimento muito ruim e demorado, ninguém responde.",
        "Muito bom, interface bonita e recursos muito úteis.",
        "Pior experiência que tive, lento e cheio de erros."
    ],
    "sentimento": [1, 1, 1, 0, 0, 0, 1, 0]  # 1=Positivo, 0=Negativo
})

# Stopwords comuns em português (preservando "não" e "nem")
stopwords_pt = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "com", "uma", "os", "no", "se", "na", "por", "mais", "as",
    "dos", "como", "mas", "ao", "ele", "das", "à", "seu", "sua",
    "ou", "quando", "muito", "nos", "já", "eu", "também", "só",
    "pelo", "pela", "até", "isso", "ela", "entre", "depois",
    "sem", "mesmo", "aos", "seus", "quem", "me", "esse", "eles",
    "você", "essa", "num", "suas", "meu", "minha", "toda", "tive"
]

# 2. Vetorização com TF-IDF (Tokenização, Limpeza e Ponderação)
vetorizador = TfidfVectorizer(
    lowercase=True,
    stop_words=stopwords_pt,
    ngram_range=(1, 1)
)

X_tfidf = vetorizador.fit_transform(dados_avaliacoes["comentario"])
y = dados_avaliacoes["sentimento"]

n_docs, n_termos = X_tfidf.shape
print(f"Matriz TF-IDF: {n_docs} avaliações x {n_termos} termos únicos")

# 3. Treinando o Modelo Classificador (Regressão Logística)
modelo_nlp = LogisticRegression()
modelo_nlp.fit(X_tfidf, y)

# 4. Testando o Modelo em Comentários Inéditos
novos_comentarios = [
    "Adorei tudo, suporte rápido e produto incrível!",
    "Serviço horrível, tudo péssimo e travando sempre."
]

novos_vetores = vetorizador.transform(novos_comentarios)
previsoes = modelo_nlp.predict(novos_vetores)
probabilidades = modelo_nlp.predict_proba(novos_vetores)

print("\n=== CLASSIFICAÇÃO DE SENTIMENTOS EM NOVOS COMENTÁRIOS ===")
for texto, pred, prob in zip(
    novos_comentarios, previsoes, probabilidades
):
    rotulo = "POSITIVO" if pred == 1 else "NEGATIVO"
    certeza = prob[pred] * 100
    print(f"\nComentário: \"{texto}\"")
    print(f"Decisão da IA: {rotulo} (Confiança: {certeza:.1f}%)")
