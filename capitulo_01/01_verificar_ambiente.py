# ====================================================================
# LIVRO: O B-A-BA da Ciência de Dados
# Capítulo 01: O Ambiente de Trabalho do Cientista de Dados
# Arquivo: 01_verificar_ambiente.py
# ====================================================================

import platform
import sys

print("=" * 60)
print(" VERIFICACAO DO AMBIENTE DE CIENCIA DE DADOS")
print("=" * 60)

# 1. Informações do Sistema e Interpretador Python
versao_py = sys.version.split()[0]
sistema = f"{platform.system()} {platform.release()}"
arquitetura = platform.architecture()[0]

print(f"[+] Versao do Python:       {versao_py}")
print(f"[+] Sistema Operacional:    {sistema} ({arquitetura})")
print("-" * 60)

# 2. Verificando Bibliotecas Essenciais (NumPy e Pandas)
for pacote in ["numpy", "pandas"]:
    rotulo = f"{pacote.capitalize():<22}"
    try:
        mod = __import__(pacote)
        print(f"[+] {rotulo} v{mod.__version__}")
    except ImportError:
        print(f"[!] {rotulo} Nao instalado")

print("-" * 60)

# 3. Testando Aceleração de Hardware Opcional (PyTorch / GPU)
try:
    import torch

    print(f"[+] PyTorch Instalado:      v{torch.__version__}")
    if torch.cuda.is_available():
        nome_gpu = torch.cuda.get_device_name(0)
        qtd_gpus = torch.cuda.device_count()
        print(f"[!] Aceleracao por GPU:     ATIVA ({qtd_gpus} GPU)")
        print(f"[!] Placa de Video:         {nome_gpu}")
    else:
        print("[*] Aceleracao por GPU:     Modo CPU ativo")
        print("    -> A CPU atende a todos os fundamentos do livro.")
except ImportError:
    print("[*] PyTorch (opcional):     Nao instalado neste ambiente.")
    print("    -> Nao se preocupe! Ele so sera usado no Modulo 6.")

print("=" * 60)
print("[OK] Diagnostico concluido com sucesso!")
print("=" * 60)
