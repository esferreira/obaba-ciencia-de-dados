# Capítulo 14 — Triagem com SLMs e Inferência Rápida via Groq
# Arquivo: 14_slms_groq_api.py

import os
import json

# 1. Modelo SLM padrão (8 bilhões de parâmetros).
# O catálogo da Groq evolui continuamente (console.groq.com/docs/models)
# e modelos maiores (como 70B) podem ser usados trocando esta constante.
MODELO_SLM = "llama-3.1-8b-instant"

# 2. Lendo a chave de API do ambiente (console.groq.com)
CHAVE_API = os.environ.get("GROQ_API_KEY")

# 3. Relato real de um cliente de comércio eletrônico
relato_cliente = (
    "Comprei um notebook gamer no dia 10/02 pelo pedido #84920. "
    "O produto chegou com a tela trincada e sem o carregador. "
    "Tentei contato no chat há 3 dias sem resposta. "
    "Quero o estorno imediato no meu cartão de crédito!"
)

# 4. Prompt do sistema exigindo saída estrita em formato JSON
prompt_sistema = (
    "Você é um analista sênior de Ciência de Dados em atendimento. "
    "Analise o relato e responda ESTRITAMENTE em formato JSON com: "
    "'sentimento' ('Positivo', 'Neutro' ou 'Negativo'), "
    "'categoria' ('Logística', 'Defeito de Produto' ou 'Suporte'), "
    "'urgencia' ('Baixa', 'Média' ou 'Crítica') e "
    "'resumo_acao' (recomendação curta para a equipe humana)."
)

# Resposta de demonstração usada apenas no modo de simulação offline
RESPOSTA_SIMULADA = {
    "sentimento": "Negativo",
    "categoria": "Defeito de Produto",
    "urgencia": "Crítica",
    "resumo_acao": (
        "Autorizar estorno imediato e agendar coleta do item avariado."
    )
}

# 5. Executando a inferência real na LPU ou simulação offline explícita
if not CHAVE_API:
    print(
    "[AVISO] GROQ_API_KEY ausente: exibindo resposta SIMULADA offline."
    )
    print(json.dumps(RESPOSTA_SIMULADA, indent=4, ensure_ascii=False))
else:
    from groq import Groq

    cliente = Groq(api_key=CHAVE_API)
    resposta = cliente.chat.completions.create(
        model=MODELO_SLM,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": relato_cliente}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    conteudo = resposta.choices[0].message.content
    resultado_json = json.loads(conteudo)
    print(f"=== TRIAGEM COM GROQ ({MODELO_SLM}) ===")
    print(json.dumps(resultado_json, indent=4, ensure_ascii=False))
