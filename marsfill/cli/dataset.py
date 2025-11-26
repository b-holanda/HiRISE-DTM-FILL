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

    def run(self, profile_name: str, execution_mode: str) -> None:
        """
        Executa o pipeline de construção do dataset baseando-se em um perfil de configuração.

        Argumentos:
            profile_name (str): Nome do perfil de configuração (ex: 'prod', 'dev') definido nos arquivos de settings.
            execution_mode (str): Define o destino dos dados ('local' ou 's3').

        Levanta:
            ValueError: Se o bucket S3 não for definido quando o modo for 's3'.
            Exception: Para erros gerais durante a execução do pipeline.
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

        download_directory: Optional[Path] = None
        s3_bucket_name: Optional[str] = None
        s3_prefix: str = build_config.get("s3_prefix", "dataset/v1/")

        if execution_mode == "local":
            output_directory_name = build_config.get("output", "dataset/v1/")
            project_root_path = Path(__file__).resolve().parent.parent.parent
            download_directory = project_root_path / output_directory_name

            logger.info(f"🔧 Modo: LOCAL. Salvando em: {download_directory}")

        elif execution_mode == "s3":
            s3_bucket_name = build_config.get("s3_bucket")

            if not s3_bucket_name:
                logger.error(
                    "Erro: Modo S3 selecionado, mas 's3_bucket' não está definido no profile."
                )
                raise ValueError("Configuração 's3_bucket' ausente para modo S3.")

            logger.info(f"🔧 Modo: S3. Bucket: {s3_bucket_name} | Prefix: {s3_prefix}")

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
                s3_bucket_name=s3_bucket_name,
                s3_prefix=s3_prefix,
                batch_size=batch_size,
                max_workers=max_workers,
            )
            builder.run()

        except Exception as error:
            logger.error(f"Falha crítica na execução do pipeline: {error}")
            raise error


def main():
    parser = argparse.ArgumentParser(description="Marsfill Dataset Builder CLI")

    parser.add_argument(
        "--profile",
        type=str,
        default="prod",
        help="Nome do perfil de configuração (ex: prod, test)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "s3"],
        required=True,
        help="Define onde os dados serão salvos: no disco local ou upload direto pro S3.",
    )

    args = parser.parse_args()

    cli_interface = DatasetCLI()
    cli_interface.run(profile_name=args.profile, execution_mode=args.mode)


if __name__ == "__main__":
    main()
