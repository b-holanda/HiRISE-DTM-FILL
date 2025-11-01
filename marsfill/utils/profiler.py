import os
import glob
import re
import yaml
import psutil

from pathlib import Path
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, NVMLError

from marsfill.utils import Logger

logger = Logger()

# --- Funções Auxiliares ---

def parse_size_to_bytes(size_str: str) -> int:
    """
    Converte uma string de tamanho (ex: "128 GB", "10 TB") para bytes.
    Assume que "GB", "TB" etc., são base 1024 (GiB, TiB).
    """
    size_str = str(size_str).strip().lower()
    units = {
        'b': 1,
        'kb': 1024, 'kib': 1024,
        'mb': 1024**2, 'mib': 1024**2,
        'gb': 1024**3, 'gib': 1024**3,
        'tb': 1024**4, 'tib': 1024**4,
    }
    
    # Regex para encontrar o número e a unidade
    match = re.match(r'^([\d\.]+)\s*(\w+)$', size_str)
    
    if not match:
        raise ValueError(f"Formato de tamanho inválido: '{size_str}'")
    
    num, unit = match.groups()
    num = float(num)
    
    if unit not in units:
        raise ValueError(f"Unidade de tamanho desconhecida: '{unit}'")
        
    return int(num * units[unit])

def get_system_hardware() -> dict:
    """
    Detecta o hardware atual do sistema e retorna um dicionário em bytes/contagens.
    """
    logger.info("🔎 Detectando hardware do sistema...")
    system_hw = {}
    
    # 1. CPU
    # Usamos núcleos físicos, que são mais relevantes para performance
    system_hw['cpu_cores'] = psutil.cpu_count(logical=False)
    
    # 2. RAM
    system_hw['ram'] = psutil.virtual_memory().total # em bytes
    
    # 3. Disco
    # Verifica o espaço total da partição raiz ("/")
    system_hw['disk'] = psutil.disk_usage('/').total # em bytes
    
    # 4. GPU (específico NVIDIA)
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        
        system_hw['gpu_platform'] = "NVIDIA"
        system_hw['gpu_memory'] = mem_info.total # em bytes
        
        # Opcional: Obter arquitetura (ex: 8 para Ampere)
        # (major, minor) = nvmlDeviceGetCudaComputeCapability(handle)
        # system_hw['gpu_arch_major'] = major 
        
        nvmlShutdown()
    except NVMLError:
        logger.info("Aviso: Não foi possível detectar GPU NVIDIA ou pynvml falhou.")
        system_hw['gpu_platform'] = "Unknown"
        system_hw['gpu_memory'] = 0
        
    logger.info("--- Hardware Detectado ---")
    logger.info(f"  CPU Cores (Físicos): {system_hw['cpu_cores']}")
    logger.info(f"  RAM Total: {system_hw['ram'] / (1024**3):.2f} GB")
    logger.info(f"  Disco Total (Raiz): {system_hw['disk'] / (1024**3):.2f} GB")
    logger.info(f"  GPU Plataforma: {system_hw['gpu_platform']}")
    # Garantir que gpu_memory seja um inteiro antes de dividir para evitar erros de tipo
    gpu_mem_bytes = system_hw.get('gpu_memory', 0)
    try:
        gpu_mem_int = int(gpu_mem_bytes)
    except (TypeError, ValueError):
        gpu_mem_int = 0
    logger.info(f"  GPU Memória: {gpu_mem_int / (1024**3):.2f} GB")
    logger.info("--------------------------\n")
    
    return system_hw

def load_profiles(profile_dir: str) -> list:
    """
    Carrega todos os arquivos *.profile.yml do diretório especificado.
    """
    profile_files = glob.glob(os.path.join(profile_dir, "*.profile.yml"))
    profiles = []
    
    logger.info(f"📂 Carregando perfis de '{profile_dir}'...")
    for f_path in profile_files:
        f_name = os.path.basename(f_path)
        try:
            with open(f_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'hardware' in config and 'hiperparameters' in config:
                    profiles.append({'name': f_name, 'config': config})
                    logger.info(f"  ✓ Carregado: {f_name}")
                else:
                    logger.info(f"  ✗ Ignorado (formato inválido): {f_name}")
        except Exception as e:
            logger.info(f"  ✗ Erro ao ler {f_name}: {e}")
            
    return profiles

def get_profile_for_hardware() -> dict | None:
    """
    Verifica o hardware do sistema e seleciona o conjunto de hiperparâmetros
    do perfil mais avançado que o sistema suporta.
    """
    profile_dir = profile_dir = os.path.join(Path(__file__).parent.parent, "profiles")

    # 1. Detectar hardware atual
    system_hw = get_system_hardware()
    
    # 2. Carregar todos os perfis
    all_profiles = load_profiles(profile_dir)
    
    valid_candidates = []
    
    logger.info("\n⚖️  Avaliando compatibilidade dos perfis...")

    for profile in all_profiles:
        try:
            reqs = profile['config']['hardware']
            reqs_gpu = reqs.get('gpu', {})
            reqs_cpu = reqs.get('cpu', {})
            req_ram = parse_size_to_bytes(reqs.get('ram', '0 B'))
            req_disk = parse_size_to_bytes(reqs.get('disk', '0 B'))
            req_cpu_cores = int(reqs_cpu.get('cores', 0))
            req_gpu_mem = parse_size_to_bytes(reqs_gpu.get('memory', '0 B'))
            req_gpu_platform = reqs_gpu.get('platform', 'Unknown').lower()

            if (system_hw['ram'] >= req_ram and
                system_hw['disk'] >= req_disk and
                system_hw['cpu_cores'] >= req_cpu_cores and
                system_hw['gpu_platform'].lower() == req_gpu_platform and
                system_hw['gpu_memory'] >= req_gpu_mem):
                
                logger.info(f"  ✓ Compatível: {profile['name']}")

                valid_candidates.append((profile, req_gpu_mem))
            else:
                 logger.info(f"  ✗ Incompatível: {profile['name']}")

        except Exception as e:
            logger.info(f"  ✗ Erro ao processar {profile['name']}: {e}")

    # 4. Selecionar o melhor candidato
    if not valid_candidates:
        logger.info("\nNenhum perfil compatível encontrado.")
        return None

    valid_candidates.sort(key=lambda item: item[1], reverse=True)

    best_profile = valid_candidates[0][0]

    logger.info(f"\n🏆 Perfil Selecionado: {best_profile['name']}\n")

    return best_profile
