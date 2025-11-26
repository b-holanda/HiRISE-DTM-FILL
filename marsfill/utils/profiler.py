import os
import glob
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from marsfill.utils import Logger

logger = Logger()

def load_all_profiles(profiles_directory_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Varre o diretório especificado em busca de arquivos com a extensão .profile.yml
    e carrega seus conteúdos em um dicionário de configurações.

    Parâmetros:
        profiles_directory_path (str): O caminho absoluto ou relativo do diretório contendo os perfis.

    Retorno:
        Dict[str, Dict[str, Any]]: Um dicionário onde a chave é o nome do perfil (sem extensão)
                                   e o valor é o conteúdo do YAML analisado.
    """
    profile_file_paths = glob.glob(os.path.join(profiles_directory_path, "*.profile.yml"))
    loaded_profiles = {}
    
    logger.info(f"📂 Carregando perfis de '{profiles_directory_path}'...")
    
    for full_file_path in profile_file_paths:
        file_name_with_extension = os.path.basename(full_file_path)
        
        try:
            with open(full_file_path, 'r', encoding='utf-8') as file_stream:
                profile_content = yaml.safe_load(file_stream)
                
                profile_key = file_name_with_extension.split(".")[0]
                loaded_profiles[profile_key] = profile_content
                
                logger.info(f"  ✓ Carregado {file_name_with_extension}")
                
        except Exception as error:
            logger.info(f"  ✗ Erro ao ler {file_name_with_extension}: {error}")
            
    return loaded_profiles

def get_profile_configuration(
    profile_name: str, 
    custom_profiles_directory: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Busca e retorna a configuração de um perfil específico.
    Calcula automaticamente o diretório padrão se nenhum for fornecido.

    Parâmetros:
        profile_name (str): O nome do perfil desejado (ex: 'prod', 'dev').
        custom_profiles_directory (Optional[str]): Caminho opcional para o diretório de perfis.
                                                   Útil para injeção de dependência em testes.

    Retorno:
        Optional[Dict[str, Any]]: O dicionário de configuração do perfil ou None se não encontrado.
    """
    if custom_profiles_directory:
        target_directory = custom_profiles_directory
    else:
        target_directory = os.path.join(Path(__file__).parent.parent, "profiles")
    
    available_profiles = load_all_profiles(target_directory)
    
    return available_profiles.get(profile_name, None)
