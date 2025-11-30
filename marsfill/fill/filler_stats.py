from pathlib import Path
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from osgeo import gdal
from marsfill.utils import Logger

logger = Logger()

class FillerStats:
    def __init__(self, output_dir: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_geotiff(self, path):
        ds = gdal.Open(str(path))
        if not ds:
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        # Fallback para nodata não definido
        if nodata is None:
            nodata = -3.4028235e38
        return array, nodata

    def calculate_metrics(self, gt_path, filled_path, mask_path=None):
        logger.info("⚡ Iniciando avaliação...")
        start_time = time.time()

        gt_arr, gt_nodata = self.load_geotiff(gt_path)
        filled_arr, _ = self.load_geotiff(filled_path)

        # 1. Definição da Máscara de Avaliação
        if mask_path:
            mask_arr, _ = self.load_geotiff(mask_path)
            # Garante que dimensões batem (crop se necessário)
            h = min(mask_arr.shape[0], gt_arr.shape[0])
            w = min(mask_arr.shape[1], gt_arr.shape[1])
            mask_arr = mask_arr[:h, :w]
            gt_arr = gt_arr[:h, :w]
            filled_arr = filled_arr[:h, :w]
            
            # Máscara binária onde o preenchimento ocorreu
            hole_mask = (mask_arr > 0)
        else:
            # Se não passar máscara, assume erro! 
            # (Em inpainting científico, avaliar a imagem toda dilui o erro)
            logger.warning("Nenhuma máscara fornecida! Avaliando imagem inteira (NÃO RECOMENDADO).")
            hole_mask = np.ones_like(gt_arr, dtype=bool)

        # 2. Validação da Máscara (CRÍTICO)
        if np.sum(hole_mask) == 0:
            logger.error("🚨 ERRO CRÍTICO: A máscara de avaliação está vazia!")
            logger.error("O input fornecido ao modelo não tinha lacunas (NoData), ou a máscara não foi salva.")
            logger.error("Métricas retornadas serão NaN para evitar vazamento de dados.")
            return self._return_empty_metrics()

        # 3. Filtragem de Pixels Inválidos no Ground Truth
        # Não podemos avaliar onde o GT original também era ruim/NoData
        valid_gt_mask = (gt_arr != gt_nodata) & np.isfinite(gt_arr) & (gt_arr > -1e30)
        
        # Pixels finais para cálculo: Onde era buraco E onde temos verdade terrestre válida
        eval_mask = hole_mask & valid_gt_mask
        
        num_pixels = int(np.sum(eval_mask))
        if num_pixels == 0:
            logger.warning("Máscara tem lacunas, mas o GT correspondente é NoData. Impossível validar.")
            return self._return_empty_metrics()

        # 4. Cálculo de Métricas
        y_true = gt_arr[eval_mask]
        y_pred = filled_arr[eval_mask]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        # SSIM (Calculado no tile inteiro, mas mascarando fora do buraco para não diluir)
        ssim_score = self._calculate_masked_ssim(gt_arr, filled_arr, eval_mask)

        metrics = {
            "rmse_m": float(rmse),
            "mae_m": float(mae),
            "ssim": float(ssim_score),
            "execution_time_s": float(time.time() - start_time),
            "evaluated_pixels": num_pixels,
        }

        self._save_metrics(metrics)
        return metrics, gt_arr, filled_arr, eval_mask

    def _calculate_masked_ssim(self, gt, pred, mask):
        """Calcula SSIM. Substitui valores fora da máscara pelo GT para focar a métrica na lacuna."""
        # Normalização Min-Max baseada no GT
        valid_vals = gt[np.isfinite(gt) & (gt > -1e30)]
        if valid_vals.size == 0: return 0.0
        mn, mx = valid_vals.min(), valid_vals.max()
        scale = mx - mn + 1e-6

        # Prepara tensores
        gt_t = torch.tensor(gt, device=self.device).unsqueeze(0).unsqueeze(0)
        pred_t = torch.tensor(pred, device=self.device).unsqueeze(0).unsqueeze(0)
        mask_t = torch.tensor(mask, device=self.device).unsqueeze(0).unsqueeze(0)

        # "Híbrido": Onde não é buraco, usamos o GT em ambas as imagens.
        # Assim, o SSIM mede apenas a degradação estrutural dentro do buraco e nas bordas.
        pred_masked = torch.where(mask_t, pred_t, gt_t)
        gt_masked = gt_t # O target é o próprio GT

        # Normaliza 0-1
        gt_norm = torch.clamp((gt_masked - mn) / scale, 0, 1)
        pred_norm = torch.clamp((pred_masked - mn) / scale, 0, 1)

        return self.ssim_calc(pred_norm, gt_norm).item()

    def _return_empty_metrics(self):
        empty = {
            "rmse_m": float("nan"), "mae_m": float("nan"), 
            "ssim": 0.0, "execution_time_s": 0.0, "evaluated_pixels": 0
        }
        self._save_metrics(empty)
        return empty, None, None, None

    def _save_metrics(self, metrics):
        json_path = self.output_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

    def plot_results(self, gt_arr, filled_arr, eval_mask, metrics, filename="result_comparison.jpg"):
        if gt_arr is None: return

        # Auto-crop para a região de interesse (onde tem máscara)
        rows = np.any(eval_mask, axis=1)
        cols = np.any(eval_mask, axis=0)
        
        if np.any(rows):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            pad = 50
            y_min = max(0, y_min - pad)
            y_max = min(gt_arr.shape[0], y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(gt_arr.shape[1], x_max + pad)
        else:
            y_min, y_max, x_min, x_max = 0, gt_arr.shape[0], 0, gt_arr.shape[1]

        gt_crop = gt_arr[y_min:y_max, x_min:x_max]
        filled_crop = filled_arr[y_min:y_max, x_min:x_max]
        mask_crop = eval_mask[y_min:y_max, x_min:x_max]

        # Configuração do Plot
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3)

        # Mapas de elevação
        vmin = np.percentile(gt_crop[np.isfinite(gt_crop)], 2)
        vmax = np.percentile(gt_crop[np.isfinite(gt_crop)], 98)

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(gt_crop, cmap="terrain", vmin=vmin, vmax=vmax)
        ax1.set_title("Ground Truth (Real)")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(filled_crop, cmap="terrain", vmin=vmin, vmax=vmax)
        ax2.set_title("Preenchimento (IA)")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Mapa de Erro
        diff = np.abs(gt_crop - filled_crop)
        diff[~mask_crop] = 0 # Só mostra erro onde preencheu
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(diff, cmap="inferno")
        ax3.set_title(f"Erro Absoluto (RMSE: {metrics.get('rmse_m', 0):.2f}m)")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Perfil Topográfico
        ax4 = fig.add_subplot(gs[1, :])
        mid_y = gt_crop.shape[0] // 2
        ax4.plot(gt_crop[mid_y, :], 'k-', lw=2, label='Ground Truth')
        ax4.plot(filled_crop[mid_y, :], 'r--', lw=1.5, label='Predição IA')
        
        # Destaca a área preenchida no gráfico
        hole_indices = np.where(mask_crop[mid_y, :])[0]
        if len(hole_indices) > 0:
            ax4.axvspan(hole_indices[0], hole_indices[-1], color='yellow', alpha=0.3, label='Lacuna Preenchida')
        
        ax4.legend()
        ax4.set_title(f"Perfil Topográfico (Transecto Central) - Amostra Local")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        logger.info(f"📊 Gráfico salvo.")
