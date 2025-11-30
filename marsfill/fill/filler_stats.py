from pathlib import Path
import time
from marsfill.utils import Logger
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from torchmetrics.image import StructuralSimilarityIndexMeasure
from osgeo import gdal

logger = Logger()


class FillerStats:
    def __init__(self, output_dir: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📂 Diretório de avaliação configurado: {self.output_dir.resolve()}")

    def load_geotiff(self, path):
        ds = gdal.Open(str(path))

        if not ds:
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()

        # Fallback: alguns arquivos trazem NoData como sentinel sem metadata
        if nodata is None and np.any(array <= -1e20):
            nodata = -3.4028234663852886e38

        return array, nodata

    def calculate_metrics(self, gt_path, filled_path, mask_path=None):
        logger.info("⚡ Iniciando avaliação...")
        start_time = time.time()

        gt_arr, gt_nodata = self.load_geotiff(gt_path)
        filled_arr, _ = self.load_geotiff(filled_path)
        # Substitui predições não finitas por GT para não descartar toda a avaliação
        non_finite_filled = ~np.isfinite(filled_arr)
        if np.any(non_finite_filled):
            filled_arr[non_finite_filled] = gt_arr[non_finite_filled]

        if mask_path:
            mask_arr, _ = self.load_geotiff(mask_path)
            # Ajusta dimensões se máscara e GT divergirem levemente
            if mask_arr.shape != gt_arr.shape:
                h = min(mask_arr.shape[0], gt_arr.shape[0])
                w = min(mask_arr.shape[1], gt_arr.shape[1])
                mask_arr = mask_arr[:h, :w]
                gt_arr = gt_arr[:h, :w]
                filled_arr = filled_arr[:h, :w]
            hole_mask = mask_arr > 0
        else:
            logger.info("⚠️ Nenhuma máscara fornecida. Avaliando na imagem inteira.")
            hole_mask = np.ones_like(gt_arr, dtype=bool)

        # Considera sentinel extremo como NoData mesmo se nodata vier None
        sentinel_mask = gt_arr <= -1e20
        if gt_nodata is None:
            gt_nodata = -3.4028234663852886e38

        finite_mask_full = np.isfinite(gt_arr) & np.isfinite(filled_arr)
        valid_gt_mask = (gt_arr != gt_nodata) & (~np.isnan(gt_arr)) & (~sentinel_mask)
        eval_mask = hole_mask & valid_gt_mask
        if np.sum(eval_mask) == 0:
            logger.warning("Máscara de avaliação vazia. Usando somente pixels válidos do GT.")
            eval_mask = valid_gt_mask

        metrics_mask = eval_mask & finite_mask_full
        if np.sum(metrics_mask) == 0:
            raise ValueError("Sem pixels válidos para avaliação após remoção de NaN/Inf.")

        y_true = gt_arr[metrics_mask]
        y_pred = filled_arr[metrics_mask]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        valid_values = gt_arr[metrics_mask]
        if valid_values.size == 0:
            raise ValueError("Sem valores válidos para normalização (GT).")
        min_val, max_val = float(valid_values.min()), float(valid_values.max())
        scale = max(max_val - min_val, 1.0)

        gt_tensor = (
            torch.tensor(gt_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        pred_tensor = (
            torch.tensor(filled_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        mask_tensor = torch.tensor(eval_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0).to(
            self.device
        )

        # Fora da região avaliada, usa o ground truth para evitar impacto na SSIM
        pred_tensor_masked = torch.where(mask_tensor, pred_tensor, gt_tensor)
        gt_tensor_masked = torch.where(mask_tensor, gt_tensor, gt_tensor)

        gt_norm = torch.clamp((gt_tensor_masked - min_val) / scale, 0.0, 1.0)
        pred_norm = torch.clamp((pred_tensor_masked - min_val) / scale, 0.0, 1.0)

        ssim_score = self.ssim_calc(pred_norm, gt_norm).item()

        end_time = time.time()
        execution_time = end_time - start_time

        metrics = {
            "rmse_m": float(rmse),
            "mae_m": float(mae),
            "ssim": float(ssim_score),
            "execution_time_s": float(execution_time),
            "evaluated_pixels": int(np.sum(eval_mask)),
        }

        json_path = self.output_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"📄 Métricas salvas em: {json_path}")

        return metrics, gt_arr, filled_arr, eval_mask

    def plot_results(
        self, gt_arr, filled_arr, eval_mask, metrics, filename="result_comparison.jpg"
    ):
        rows = np.any(eval_mask, axis=1)
        cols = np.any(eval_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            y_min, y_max, x_min, x_max = 0, gt_arr.shape[0], 0, gt_arr.shape[1]
        else:
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

        pad = 50
        y_min = max(0, y_min - pad)
        y_max = min(gt_arr.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(gt_arr.shape[1], x_max + pad)

        gt_crop = gt_arr[y_min:y_max, x_min:x_max]
        filled_crop = filled_arr[y_min:y_max, x_min:x_max]
        mask_crop = eval_mask[y_min:y_max, x_min:x_max]
        diff_map = np.abs(gt_crop - filled_crop)
        diff_map[~mask_crop] = 0

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        finite_gt = np.isfinite(gt_crop) & (np.abs(gt_crop) < 1e19)
        if not np.any(finite_gt):
            finite_gt = np.ones_like(gt_crop, dtype=bool)
        gt_vals = gt_crop[finite_gt]
        vmin_gt, vmax_gt = np.percentile(gt_vals, (2, 98))
        gt_crop = np.clip(gt_crop, vmin_gt, vmax_gt)

        im1 = ax1.imshow(gt_crop, cmap="terrain", vmin=vmin_gt, vmax=vmax_gt)
        ax1.set_title("Ground Truth (Real)")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        finite_filled = np.isfinite(filled_crop) & (np.abs(filled_crop) < 1e19)
        if not np.any(finite_filled):
            finite_filled = np.ones_like(filled_crop, dtype=bool)
        filled_vals = filled_crop[finite_filled]
        vmin_fill, vmax_fill = np.percentile(filled_vals, (2, 98))
        filled_crop = np.clip(filled_crop, vmin_fill, vmax_fill)
        im2 = ax2.imshow(filled_crop, cmap="terrain", vmin=vmin_fill, vmax=vmax_fill)
        ax2.set_title("Preenchimento (IA)")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        diff_vals = diff_map[mask_crop] if np.any(mask_crop) else diff_map[np.isfinite(diff_map)]
        vmax_diff = np.percentile(diff_vals, 95) if diff_vals.size > 0 else diff_map.max(initial=0)
        im3 = ax3.imshow(diff_map, cmap="magma", vmin=0, vmax=max(vmax_diff, 1e-6))
        ax3.set_title("Erro Absoluto")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Metros")

        ax4 = fig.add_subplot(gs[1, :])
        mid_y = gt_crop.shape[0] // 2

        ax4.plot(gt_crop[mid_y, :], "k-", linewidth=2, label="Real", alpha=0.7)
        ax4.plot(filled_crop[mid_y, :], "r--", linewidth=1.5, label="IA")

        hole_indices = np.where(mask_crop[mid_y, :])[0]
        if len(hole_indices) > 0:
            ax4.axvspan(
                hole_indices[0],
                hole_indices[-1],
                color="yellow",
                alpha=0.2,
                label="Área Preenchida",
            )

        ax4.set_title(f"Perfil Topográfico (Transecto Y={mid_y + y_min})")
        ax4.set_ylabel("Elevação (m)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"📊 Gráfico salvo em: {output_path}")
