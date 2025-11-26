
import argparse
from logging import Logger
import os
from pathlib import Path

from marsfill.fill.dtm_filler import DTMFiller
from marsfill.fill.filler_stats import FillerStats
from marsfill.model.eval import Evaluator
from marsfill.model.train import AvaliableModels
from marsfill.utils.profiler import get_profile

logger = Logger()

def main():
    parser = argparse.ArgumentParser(
                    prog='Mars DTM Fill',
                    description='CLI para preenchimento de lacunas em DTMs',
                    epilog='')
    
    parser.add_argument('-dtm', '--dtm', help='caminho completo para DTM com lacunas a serem preenchidas')
    parser.add_argument('-ortho', '--ortho', help='caminho completo para orthoimagem que será utilizada para preencher as lacunas do dtm')
    parser.add_argument('-p', '--profile', help='Perfil que executará o script [prod] ou [test]')

    args = parser.parse_args()
    
    if not args.dtm:
        raise argparse.ArgumentError('Caminho para DTM é inválido')
    
    if not args.ortho:
        raise argparse.ArgumentError('Caminho para orthoimagem é inválido')
    
    dtm_path = Path(args.dtm)
    ortho_path = Path(args.ortho)
    output_folder = Path(__name__).parent.parent / "outputs" / os.path.basename(dtm_path).split('.')[0].lower()

    profile_name = "prod"
    profile = get_profile(profile_name)

    model_path = Path(__name__).parent.parent / str(profile["fill"]["model_path"])
    selected_model = AvaliableModels.INTEL_DPT_LARGE

    padding_size = int(profile["fill"]["padding_size"])
    tile_size = int(profile["fill"]["tile_size"])

    filler = DTMFiller(
        evaluator=Evaluator(model_path=model_path, pretrained_model_name=selected_model),
        padding_size=padding_size,
        tile_size=tile_size
    )

    filled_dtm_path, filled_dtm_mask_path = filler.fill(dtm_path=dtm_path, ortho_path=ortho_path, output_folder=output_folder)

    stats = FillerStats(output_dir=output_folder)

    metrics, gt_arr, filled_arr, eval_mask = stats.calculate_metrics(
        gt_path=dtm_path,
        filled_path=filled_dtm_path,
        mask_path=filled_dtm_mask_path
    )

    stats.plot_results(eval_mask=eval_mask, filled_arr=filled_arr, gt_arr=gt_arr, metrics=metrics)

if __name__ == '__main__':
    main()