# ==============================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 01: Verificação do Ambiente e GPU
# Arquivo: 01_verificar_ambiente.py
# ==============================================================================
# Repositório Oficial de Códigos do Livro
# Execute este arquivo localmente no VS Code ou copie para o Google Colab.
# ==============================================================================

import sys
import platform

print("=" * 60)
print(" VERIFICACAO DO AMBIENTE DE CIENCIA DE DADOS")
print("=" * 60)

# 1. Informações do Sistema e Interpretador Python
print(f"[+] Versao do Python:       {sys.version.split()[0]}")
print(f"[+] Sistema Operacional:    {platform.system()} {platform.release()} ({platform.architecture()[0]})")
print("-" * 60)

# 2. Testando Bibliotecas e Aceleração de Hardware (GPU / CUDA)
try:
    import torch
    print(f"[+] PyTorch Instalado:      v{torch.__version__}")
    
    tem_gpu = torch.cuda.is_available()
    if tem_gpu:
        nome_gpu = torch.cuda.get_device_name(0)
        qtd_gpus = torch.cuda.device_count()
        print(f"[!] Aceleracao por GPU:     ATIVA ({qtd_gpus} dispositivo(s) detectado(s))")
        print(f"[!] Placa de Video:         {nome_gpu}")
    else:
        print("[*] Aceleracao por GPU:     Nao detectada (Modo CPU ativo)")
        print("    -> Tudo pronto! A CPU e ideal para todos os fundamentos do livro.")
        
except ImportError:
    print("[!] PyTorch nao encontrado neste ambiente.")
    print("    Para instalar no seu computador local, execute: pip install torch")

print("=" * 60)
print("[OK] Teste de ambiente concluido com sucesso!")
print("=" * 60)
