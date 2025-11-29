import argparse
import sys
from pathlib import Path
from typing import Optional

from marsfill.dataset.build import DatasetBuilder
from marsfill.utils import Logger
from marsfill.utils.profiler import get_profile

logger = Logger()


class DatasetCLI:
    """Interface de Linha de Comando (CLI) para orquestrar a construção do dataset Marsfill."""

    def run(self, profile_name: str) -> None:
        """
        Executa o pipeline de construção do dataset a partir de um perfil.

        Args:
            profile_name: Nome do perfil de configuração (ex.: `prod`, `dev`).

        Raises:
            Exception: Para qualquer falha no pipeline.
        """
        configuration_profile = get_profile(profile_name)
        if not configuration_profile:
            logger.error(f"Perfil '{profile_name}' não encontrado.")
            sys.exit(1)

        build_config = configuration_profile.get("make", {})

        total_samples: int = build_config.get("samples", 100)
        urls_to_scan: list[str] = build_config.get(
            "urls_to_scan",
            ["https://www.uahirise.org/PDS/DTM/PSP/", "https://www.uahirise.org/PDS/DTM/ESP/"],
        )
        tile_size: int = build_config.get("tile_size", 512)
        stride_size: int = build_config.get("stride", 256)
        batch_size: int = build_config.get("batch_size", 500)
        max_workers: Optional[int] = build_config.get("max_workers", None)

        output_directory_name = build_config.get("output", "data/dataset/v1/")
        project_root_path = Path(__file__).resolve().parent.parent.parent
        download_directory = project_root_path / output_directory_name

        logger.info(f"🔧 Modo: LOCAL. Salvando em: {download_directory}")

        logger.info("--- Parâmetros do Dataset ---")
        logger.info(f"Samples: {total_samples}")
        logger.info(f"Tile Size: {tile_size}")
        logger.info(f"Stride Size: {stride_size}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Workers: {max_workers if max_workers else 'AUTO'}")

        try:
            builder = DatasetBuilder(
                urls_to_scan=urls_to_scan,
                total_samples=total_samples,
                tile_size=tile_size,
                stride_size=stride_size,
                download_directory=download_directory,
                batch_size=batch_size,
                max_workers=max_workers,
            )
            builder.run()

        except Exception as error:
            logger.error(f"Falha crítica na execução do pipeline: {error}")
            raise error


def main():
    """Ponto de entrada da CLI de dataset."""
    parser = argparse.ArgumentParser(description="Marsfill Dataset Builder CLI")

    parser.add_argument(
        "--profile",
        type=str,
        default="prod",
        help="Nome do perfil de configuração (ex: prod, test)",
    )

    args = parser.parse_args()

    cli_interface = DatasetCLI()
    cli_interface.run(profile_name=args.profile)


if __name__ == "__main__":
    main()
