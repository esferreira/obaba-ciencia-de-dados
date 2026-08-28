# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 14: Modelos de Linguagem na Prática via API Groq
# Arquivo: 14_slms_groq_api.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------import os
import json
from groq import Groq

# 1. Configurando a Chave de API do Groq (Obtida gratuitamente em console.groq.com)
# Em produção, use sempre variáveis de ambiente: os.environ["GROQ_API_KEY"]
CHAVE_API = os.environ.get("GROQ_API_KEY", "SUA_CHAVE_AQUI")

# 2. Inicializando o Cliente Groq
cliente = Groq(api_key=CHAVE_API)

# 3. Texto real de um cliente de e-commerce
relato_cliente = """
Comprei um notebook gamer no dia 10/02 pelo pedido #84920. O produto chegou com a tela trincada 
e o carregador faltando na caixa. Tentei contato pelo chat e ninguém me respondeu há 3 dias. 
Estou extremamente frustrado e quero o estorno imediato no meu cartão de crédito!
"""

# 4. Construindo o Prompt do Cientista de Dados (Instruindo a IA a agir como analista)
prompt_sistema = """
Você é um assistente sênior de Ciência de Dados especializado em triagem de atendimento ao cliente.
Analise o relato do cliente e responda ESTRITAMENTE em formato JSON com as seguintes chaves:
- "sentimento": "Positivo", "Neutro" ou "Negativo"
- "categoria": "Logística", "Defeito de Produto", "Financeiro" ou "Suporte"
- "urgencia": "Baixa", "Média" ou "Crítica"
- "resumo_acao": Uma frase curta com a recomendação imediata para a equipe humana.
"""

# 5. Executando a Inferência Ultra-Rápida com Llama 3.3 70B no Groq
try:
    resposta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": relato_cliente}
        ],
        temperature=0.1,  # Temperatura baixa para respostas determinísticas e precisas
        response_format={"type": "json_object"}  # Força o retorno em JSON estruturado
    )
    
    # 6. Exibindo o Resultado Analítico Estruturado
    resultado_json = json.loads(resposta.choices[0].message.content)
    print("=== TRIAGEM AUTOMATIZADA COM IA / GROQ ===")
    print(json.dumps(resultado_json, indent=4, ensure_ascii=False))
    
except Exception as e:
    print("Para executar a chamada real, configure sua chave gratuita em console.groq.com!")
    print(f"Exemplo de Retorno Simulado:\n{{'sentimento': 'Negativo', 'categoria': 'Defeito de Produto', 'urgencia': 'Crítica', 'resumo_acao': 'Autorizar estorno imediato e solicitar recolhimento do item avariado.'}}")
