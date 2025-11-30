import argparse
import sys
from pathlib import Path
from marsfill.fill.dtm_filler import DTMFiller
from marsfill.model.eval import Evaluator
from marsfill.model.train import AvailableModels
from marsfill.utils import Logger
from marsfill.fill.filler_stats import FillerStats
# Se o arquivo filler_stats.py estiver em marsfill/fill/, use:
# from marsfill.fill.filler_stats import FillerStats

logger = Logger()

def parse_args():
    parser = argparse.ArgumentParser(description="MarsFill - DTM Inpainting Tool")
    
    parser.add_argument("--profile", type=str, default="prod", help="Perfil de execução")
    parser.add_argument("--dtm", type=str, required=True, help="Caminho do DTM de entrada (com lacunas)")
    parser.add_argument("--ortho", type=str, required=True, help="Caminho da Ortoimagem")
    parser.add_argument("--gt", type=str, required=False, help="Caminho do Ground Truth")
    parser.add_argument("--out_dir", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--model_path", type=str, default="data/models/marsfill_model.pth", help="Caminho do modelo")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    dtm_path = Path(args.dtm)
    ortho_path = Path(args.ortho)
    out_dir = Path(args.out_dir)
    model_path = Path(args.model_path)
    gt_path = Path(args.gt) if args.gt else dtm_path

    if not dtm_path.exists():
        logger.error(f"DTM não encontrado: {dtm_path}")
        sys.exit(1)

    try:
        # Inicialização
        # Nota: Ajuste AvailableModels.DPT_LARGE conforme sua enumeração real em train.py
        evaluator = Evaluator(pretrained_model_name=AvailableModels.INTEL_DPT_LARGE, model_path_uri=str(model_path))
        
        filler = DTMFiller(evaluator=evaluator, padding_size=128, tile_size=512)
        
        # Execução
        final_dtm, final_mask, _, _, _ = filler.fill(
            dtm_path=dtm_path,
            ortho_path=ortho_path,
            output_root=str(out_dir)
        )

        stats = FillerStats(output_dir=out_dir)
        
        logger.info("⚡ Calculando estatísticas e gerando gráficos...")
        
        metrics, gt_arr, filled_arr, mask_arr = stats.calculate_metrics(
            gt_path=gt_path,
            filled_path=final_dtm,
            mask_path=final_mask
        )
        
        if metrics['evaluated_pixels'] > 0:
            logger.info(f"RMSE: {metrics['rmse_m']:.4f} m | SSIM: {metrics['ssim']:.4f}")
            
            # --- NOVA CHAMADA PARA GERAR TODOS OS ARQUIVOS ---
            stats.generate_all_outputs(
                gt_path=gt_path,
                input_path=dtm_path,   # Passa o DTM com buracos
                ortho_path=ortho_path, # Passa a Ortoimagem
                filled_path=final_dtm,
                mask_path=final_mask,
                metrics=metrics
            )
            logger.info("✅ Gráficos e imagens gerados com sucesso.")
        else:
            logger.warning("Validação inválida (sem pixels avaliados).")

    except Exception as e:
        logger.error(f"Falha crítica: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
