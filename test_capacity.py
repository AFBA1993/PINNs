import torch
import platform
import psutil

def check_research_power():
    print(f"--- 🖥️ Diagnóstico de Hardware para PINNs ---")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"RAM Total: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    
    # Verificação de GPU (O motor das PINNs)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_capability = torch.cuda.get_device_capability(0)
        print(f"✅ GPU Detectada: {device_name}")
        print(f"   VRAM Disponível: {round(vram, 2)} GB")
        print(f"   Compute Capability: {cuda_capability[0]}.{cuda_capability[1]}")
    elif torch.backends.mps.is_available():
        print("✅ Apple Silicon (MPS) Detectado - Bom para prototipagem 1D.")
    else:
        print("⚠️ Nenhuma GPU detectada. PINNs complexas (Euler/2D) serão lentas.")

    # Verificação de Precisão (Essencial para PoF/IJHMT)
    tensor_test = torch.zeros(1).to(torch.float64)
    print(f"--- 🧪 Teste de Precisão ---")
    print(f"Suporte a Float64 (Double Precision): {'Sim' if tensor_test.dtype == torch.float64 else 'Não'}")

check_research_power()