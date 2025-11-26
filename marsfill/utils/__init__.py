import glob
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import Optional, List, Tuple, Any
import s3fs

class CandidateType(Enum):
    DIGITAL_TERRAIN_MODEL = 1
    ORTHOPHOTO = 2

@dataclass(frozen=True)
class CandidateFile:
    filename: str
    url_reference: str
    file_type: CandidateType

class ApplicationLogger:
    """
    Singleton para gerenciamento de logs da aplicação.
    Garante que apenas uma instância de configuração de log exista.
    """
    _instance = None

    def __new__(cls, *arguments, **keyword_arguments):
        """
        Cria ou retorna a instância única da classe.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_file_path: Optional[Path] = None) -> None:
        """
        Inicializa o logger. Configura handlers de console e arquivo se ainda não inicializado.

        Parâmetros:
            log_file_path (Optional[Path]): Caminho para o arquivo de log. Se None, usa 'app.log' no diretório atual.
        """
        if hasattr(self, '_initialized'):
            return

        self._internal_logger = logging.getLogger("marsfill")
        self._internal_logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self._internal_logger.addHandler(console_handler)

        if log_file_path is None:
            log_file_path = Path("app.log")

        try:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            self._internal_logger.addHandler(file_handler)
        except IOError:
            print(f"Aviso: Não foi possível criar arquivo de log em {log_file_path}. Apenas logs de console estarão ativos.")

        self._initialized = True

    def info(self, message: str) -> None:
        """
        Registra uma mensagem de nível INFO.

        Parâmetros:
            message (str): A mensagem a ser logada.
        """
        self._internal_logger.info(message)

    def debug(self, message: str) -> None:
        """
        Registra uma mensagem de nível DEBUG.

        Parâmetros:
            message (str): A mensagem a ser logada.
        """
        self._internal_logger.debug(message)

    def error(self, message: str, exception_info: bool = True) -> None:
        """
        Registra uma mensagem de nível ERROR.

        Parâmetros:
            message (str): A mensagem de erro.
            exception_info (bool): Se True, inclui a stack trace da exceção.
        """
        self._internal_logger.error(message, stack_info=True, exc_info=exception_info)

def generate_expected_filenames(hirise_id: str) -> Tuple[str, str]:
    """
    Gera os nomes de arquivos esperados para um dado ID HiRISe.

    Parâmetros:
        hirise_id (str): O identificador da captura HiRISe (ex: ESP_012345_1234).

    Retorno:
        Tuple[str, str]: Uma tupla contendo (nome_arquivo_tif_padrao, nome_arquivo_ortho_especifico).
    """
    hirise_id_parts = hirise_id.split("_")
    return f"{hirise_id}.tif", f"ESP_{hirise_id_parts[1]}_{hirise_id_parts[2]}_RED_B_01_ORTHO.tif"

def find_digital_terrain_model_candidate(candidate_files_list: List[CandidateFile]) -> Optional[CandidateFile]:
    """
    Busca um arquivo do tipo DIGITAL_TERRAIN_MODEL na lista fornecida.

    Parâmetros:
        candidate_files_list (List[CandidateFile]): Lista de candidatos a arquivo.

    Retorno:
        Optional[CandidateFile]: O objeto CandidateFile se encontrado, caso contrário None.
    """
    return find_candidate_by_type(candidate_files_list, CandidateType.DIGITAL_TERRAIN_MODEL)

def find_orthophoto_candidate(candidate_files_list: List[CandidateFile]) -> Optional[CandidateFile]:
    """
    Busca um arquivo do tipo ORTHOPHOTO na lista fornecida.

    Parâmetros:
        candidate_files_list (List[CandidateFile]): Lista de candidatos a arquivo.

    Retorno:
        Optional[CandidateFile]: O objeto CandidateFile se encontrado, caso contrário None.
    """
    return find_candidate_by_type(candidate_files_list, CandidateType.ORTHOPHOTO)

def find_candidate_by_type(candidate_files_list: List[CandidateFile], target_type: CandidateType) -> Optional[CandidateFile]:
    """
    Função auxiliar genérica para filtrar candidatos por tipo.

    Parâmetros:
        candidate_files_list (List[CandidateFile]): Lista de arquivos candidatos.
        target_type (CandidateType): O tipo de arquivo desejado (DTM ou ORTHO).

    Retorno:
        Optional[CandidateFile]: O primeiro candidato correspondente encontrado ou None.
    """
    return next((item for item in candidate_files_list if item.file_type == target_type), None)

def load_dataset_files(base_directory: Path, orthophoto_pattern: str, digital_terrain_model_pattern: str) -> Tuple[List[str], List[str]]:
    """
    Busca arquivos no sistema de arquivos local que correspondam aos padrões fornecidos.

    Parâmetros:
        base_directory (Path): O diretório base para a busca.
        orthophoto_pattern (str): Padrão glob para arquivos de ortofoto (ex: '*ORTHO.tif').
        digital_terrain_model_pattern (str): Padrão glob para arquivos DTM (ex: '*DTM.tif').

    Retorno:
        Tuple[List[str], List[str]]: Duas listas contendo caminhos de arquivos encontrados (ortho, dtm).
                                     Retorna listas vazias se nada for encontrado.
    """
    logger = ApplicationLogger()
    
    found_ortho_files = sorted(glob.glob(os.path.join(base_directory, orthophoto_pattern)))
    if not found_ortho_files:
        logger.error(f"Nenhum arquivo '{orthophoto_pattern}' encontrado em {base_directory}")

    found_dtm_files = sorted(glob.glob(os.path.join(base_directory, digital_terrain_model_pattern)))
    if not found_dtm_files:
        logger.error(f"Nenhum arquivo '{digital_terrain_model_pattern}' encontrado em {base_directory}")

    return found_ortho_files, found_dtm_files

def validate_dataset_pairs(ortho_files_list: List[str], dtm_files_list: List[str]) -> Tuple[List[str], List[str]]:
    """
    Valida se para cada ortofoto existe um DTM correspondente e se as listas não estão vazias.

    Parâmetros:
        ortho_files_list (List[str]): Lista de arquivos de ortofoto.
        dtm_files_list (List[str]): Lista de arquivos de DTM.

    Retorno:
        Tuple[List[str], List[str]]: As mesmas listas, se passarem na validação.

    Levanta:
        ValueError: Se alguma lista estiver vazia.
        FileNotFoundError: Se um arquivo DTM esperado não existir fisicamente.
    """
    logger = ApplicationLogger()

    if not ortho_files_list:
        raise ValueError("Arquivos de orthoimagens não encontrados ou lista vazia.")

    if not dtm_files_list:
        raise ValueError("Arquivos de DTM não encontrados ou lista vazia.")

    for orthophoto_path, dtm_path in zip(ortho_files_list, dtm_files_list):
        if not os.path.exists(dtm_path):
            logger.error(f"Arquivo DTM correspondente {dtm_path} não encontrado para {orthophoto_path}")
            raise FileNotFoundError(f"Arquivo não encontrado: {dtm_path}")

    return ortho_files_list, dtm_files_list

def convert_strings_to_paths(path_strings: List[str]) -> List[Path]:
    """
    Converte uma lista de strings em uma lista de objetos Path.

    Parâmetros:
        path_strings (List[str]): Lista de caminhos em formato string.

    Retorno:
        List[Path]: Lista de objetos Path correspondentes.
    """
    return [Path(path_string) for path_string in path_strings]

def list_parquet_files(directory_path: str, file_system_client: Optional[Any] = None) -> List[str]:
    """
    Lista arquivos Parquet recursivamente, suportando S3 e disco local.
    Permite injeção de dependência do sistema de arquivos para testes.

    Parâmetros:
        directory_path (str): Caminho local ou URI S3 (s3://bucket/path).
        file_system_client (Optional[Any]): Instância de cliente de sistema de arquivos (ex: s3fs). 
                                            Se None e for S3, cria um novo cliente.

    Retorno:
        List[str]: Lista ordenada de caminhos completos dos arquivos encontrados.
    """
    normalized_directory = str(directory_path)
    found_files = []

    if normalized_directory.startswith("s3://"):
        if file_system_client is None:
            file_system_interface = s3fs.S3FileSystem(anon=False)
        else:
            file_system_interface = file_system_client
        
        found_paths = file_system_interface.glob(f"{normalized_directory}/**/*.parquet")
        found_files = [f"s3://{path}" for path in found_paths]

    else:
        path_object = Path(normalized_directory)
        if not path_object.exists():
            return []
            
        found_paths_generator = path_object.rglob("*.parquet")
        found_files = [str(path) for path in found_paths_generator]

    return sorted(found_files)
