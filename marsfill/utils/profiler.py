import os
import glob
import yaml

from pathlib import Path

from marsfill.utils import Logger

logger = Logger()

def load_profiles(profile_dir: str) -> dict[str, dict]:
    """
    Carrega todos os arquivos *.profile.yml do diretório especificado.
    """
    profile_files = glob.glob(os.path.join(profile_dir, "*.profile.yml"))
    profiles = {}
    
    logger.info(f"📂 Carregando perfis de '{profile_dir}'...")
    for f_path in profile_files:
        f_name = os.path.basename(f_path)
        try:
            with open(f_path, 'r') as f:
                config = yaml.safe_load(f)
                profiles[f_name.split(".")[0]] = config
                
                logger.info(f"  ✓ Carregado {f_name}")
        except Exception as e:
            logger.info(f"  ✗ Erro ao ler {f_name}: {e}")
            
    return profiles

def get_profile(profile: str) -> dict | None:
    """
    Verifica o hardware do sistema e seleciona o conjunto de hiperparâmetros
    do perfil mais avançado que o sistema suporta.
    """
    profile_dir = profile_dir = os.path.join(Path(__file__).parent.parent, "profiles")
    
    return load_profiles(profile_dir).get(profile, None)
