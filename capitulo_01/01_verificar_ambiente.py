# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 01: Verificação do Ambiente e GPU
# Arquivo: 01_verificar_ambiente.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

# ----------------------------------------------------------------------------
# PRÓXIMO BLOCO DE CÓDIGO DO CAPÍTULO
# ----------------------------------------------------------------------------import sys
import platform

print("=" * 60)
print(" 🔬 VERIFICAÇÃO DO AMBIENTE DE CIÊNCIA DE DADOS")
print("=" * 60)

# 1. Informações do Sistema e Interpretador Python
print(f"🐍 Versão do Python:       {sys.version.split()[0]}")
print(f"💻 Sistema Operacional:    {platform.system()} {platform.release()} ({platform.architecture()[0]})")
print("-" * 60)

# 2. Testando Bibliotecas e Aceleração de Hardware (GPU / CUDA)
try:
    import torch
    print(f"📦 PyTorch Instalado:      v{torch.__version__}")
    
    tem_gpu = torch.cuda.is_available()
    if tem_gpu:
        nome_gpu = torch.cuda.get_device_name(0)
        qtd_gpus = torch.cuda.device_count()
        print(f"🔥 Aceleração por GPU:     ATIVA ({qtd_gpus} dispositivo(s) detectado(s))")
        print(f"🚀 Placa de Vídeo:         {nome_gpu}")
    else:
        print("💡 Aceleração por GPU:     Não detectada (Modo CPU ativo)")
        print("   -> Tudo pronto! A CPU é ideal para todos os fundamentos do livro.")
        
except ImportError:
    print("⚠️  PyTorch não encontrado neste ambiente.")
    print("   Para instalar no seu computador local, execute: pip install torch")

print("=" * 60)
print("✅ Teste de ambiente concluído com sucesso!")
print("=" * 60)
