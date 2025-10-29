import requests
import re
from dataclasses import dataclass
from typing import List, Optional, Iterable, Set
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from marsfill.utils import Logger

logger = Logger()

@dataclass(frozen=True)
class ProductPair:
    """Armazena um par de URLs de DTM e Orthoimage."""
    dtm_url: str
    ortho_url: str
    
    @property
    def id(self) -> str:
        match = re.search(r"(ESP|PSP)_\d{6}_\d{4}", self.ortho_url)
        return match.group(0) if match else "unknown_id"

class HirisePDSIndexerDFS:
    """
    Navega pelo repositório PDS da HiRISE usando uma abordagem de
    Busca em Profundidade (DFS) para indexar pares de DTM e Ortho.
    """

    def __init__(self, base_urls: Iterable[str]):
        """
        Inicializa o indexador DFS.

        Args:
            base_urls: Uma lista ou iterável de URLs raiz para começar a varredura
                         (ex: [.../DTM/PSP/, .../DTM/ESP/]).
        """
        self.start_urls = list(base_urls)
        self.session = requests.Session()

        self.dtm_pattern = re.compile(
            r"DTEEC_\d{6}_\d{4}_\d{6}_\d{4}_[A-Z]\d{2}\.IMG$", 
            re.IGNORECASE
        )

        self.ortho_pattern = re.compile(
            r"(ESP|PSP)_\d{6}_\d{4}_RED_A_\d{2}_ORTHO\.JP2$", 
            re.IGNORECASE
        )

        self.pairs_found: List[ProductPair] = []
        self.visited_dirs: Set[str] = set()

    def _fetch_and_parse(self, url: str) -> tuple[List[str], List[str]]:
        """
        Busca o HTML de uma URL e retorna duas listas:
        1. Links de subdiretórios
        2. Links de arquivos
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Falha ao buscar {url}: {e}")
            return [], []

        soup = BeautifulSoup(response.text, 'html.parser')
        dir_links: List[str] = []
        file_links: List[str] = []

        pre_tag = soup.find('pre')
        if not pre_tag:
            logger.info(f"Nenhuma tag <pre> encontrada em {url}, pulando.")
            return [], []

        for link in pre_tag.find_all('a'):
            href = str(link.get('href') or "")

            if not href or href == '../' or href.startswith('?'):
                continue

            absolute_url = urljoin(url, href)

            if href.endswith('/'):
                if absolute_url not in self.visited_dirs:
                    dir_links.append(absolute_url)
            else:
                file_links.append(absolute_url)

        return dir_links, file_links

    def _process_leaf_page(self, file_urls: List[str]) -> bool:
        """
        Processa uma página que contém arquivos (um "nó folha").
        Tenta encontrar um par DTM/ORTHO nela.
        """
        dtm_url = None
        ortho_url = None
        
        for file_url in file_urls:
            filename = file_url.split('/')[-1]
            
            if not dtm_url and self.dtm_pattern.search(filename):
                dtm_url = file_url
            
            if not ortho_url and self.ortho_pattern.search(filename):
                ortho_url = file_url

            if dtm_url and ortho_url:
                break

        if dtm_url and ortho_url:
            pair = ProductPair(dtm_url=dtm_url, ortho_url=ortho_url)
            logger.info(f"Par encontrado ({pair.id})")
            self.pairs_found.append(pair)
            return True

        return False

    def index_pairs(self, max_pairs: Optional[int] = None) -> List[ProductPair]:
        """
        Varre o PDS usando DFS iterativo e indexa os pares de DTM/Orthoimage.

        Args:
            max_pairs: O número máximo de pares para encontrar. 
                         Se None, busca todos.

        Returns:
            Uma lista de objetos ProductPair.
        """
        self.pairs_found = []
        self.visited_dirs = set()

        stack: List[str] = list(self.start_urls)
        
        logger.info(f"Iniciando varredura DFS em {len(stack)} URLs base...")

        while stack:
            if max_pairs is not None and len(self.pairs_found) >= max_pairs:
                logger.info(f"Limite de {max_pairs} pares atingido. Parando a varredura.")
                break

            current_url = stack.pop()
            
            if current_url in self.visited_dirs:
                continue

            self.visited_dirs.add(current_url)
            logger.debug(f"Varrendo: {current_url}")

            dir_links, file_links = self._fetch_and_parse(current_url)

            if file_links:
                found_pair = self._process_leaf_page(file_links)
                
                if found_pair:
                    continue

            if dir_links:
                logger.debug(f"Encontrados {len(dir_links)} subdiretórios. Adicionando à pilha.")
                for dir_url in reversed(dir_links):
                    if dir_url not in self.visited_dirs:
                        stack.append(dir_url)

        logger.info(f"Varredura DFS concluída. Total de pares encontrados: {len(self.pairs_found)}")
        return self.pairs_found
