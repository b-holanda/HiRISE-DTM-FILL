import gc
import json
import time
from pathlib import Path

import matplotlib
# Configura backend 'Agg' para velocidade e headless servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import torch
from osgeo import gdal
from torchmetrics.image import StructuralSimilarityIndexMeasure

from marsfill.utils import Logger

logger = Logger()

class FillerStats:
    def __init__(self, output_dir: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_geotiff(self, path):
        """Carrega GeoTIFF e retorna array numpy."""
        ds = gdal.Open(str(path))
        if not ds:
            return None, None
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()

        if nodata is None:
            if np.nanmin(array) < -1e30:
                nodata = -3.4028235e38
            else:
                nodata = None
        
        ds = None 
        return array, nodata

    def calculate_metrics(self, gt_path, filled_path, mask_path=None):
        logger.info("⚡ Calculando métricas...")
        start_time = time.time()

        gt_arr, gt_nodata = self.load_geotiff(gt_path)
        filled_arr, _ = self.load_geotiff(filled_path)

        if gt_arr is None or filled_arr is None:
            return self._return_empty_metrics()

        if mask_path:
            mask_arr, _ = self.load_geotiff(mask_path)
            h = min(mask_arr.shape[0], gt_arr.shape[0])
            w = min(mask_arr.shape[1], gt_arr.shape[1])
            mask_arr = mask_arr[:h, :w]
            hole_mask = (mask_arr > 0)
            del mask_arr 
        else:
            hole_mask = np.ones_like(gt_arr, dtype=bool)

        h, w = hole_mask.shape
        gt_arr = gt_arr[:h, :w]
        filled_arr = filled_arr[:h, :w]

        if np.sum(hole_mask) == 0:
            return self._return_empty_metrics()

        valid_gt_mask = np.isfinite(gt_arr) & (gt_arr > -1e30)
        if gt_nodata is not None:
            valid_gt_mask &= ~np.isclose(gt_arr, gt_nodata)

        eval_mask = hole_mask & valid_gt_mask
        del hole_mask, valid_gt_mask
        
        num_pixels = int(np.sum(eval_mask))
        if num_pixels == 0:
            return self._return_empty_metrics()

        y_true = gt_arr[eval_mask]
        y_pred = filled_arr[eval_mask]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        del y_true, y_pred

        ssim_score = self._calculate_masked_ssim(gt_arr, filled_arr, eval_mask)

        del gt_arr, filled_arr, eval_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = {
            "rmse_m": float(rmse),
            "mae_m": float(mae),
            "ssim": float(ssim_score),
            "execution_time_s": float(time.time() - start_time),
            "evaluated_pixels": num_pixels,
        }

        self._save_metrics(metrics)
        return metrics

    def generate_all_outputs(self, gt_path, input_path, ortho_path, filled_path, mask_path, metrics):
        """
        Gera imagens com downsampling para velocidade.
        """
        logger.info("🎨 Gerando imagens (Fast Mode)...")

        # 1. Determina Crop e Fator de Decimação
        mask_arr, _ = self.load_geotiff(mask_path)
        if mask_arr is None: return

        rows = np.any(mask_arr > 0, axis=1)
        cols = np.any(mask_arr > 0, axis=0)
        del mask_arr
        
        if np.any(rows):
            ds_gt = gdal.Open(str(gt_path))
            max_h, max_w = ds_gt.RasterYSize, ds_gt.RasterXSize
            ds_gt = None

            pad = 50
            y_min = max(0, np.where(rows)[0][0] - pad)
            y_max = min(max_h, np.where(rows)[0][-1] + pad)
            x_min = max(0, np.where(cols)[0][0] - pad)
            x_max = min(max_w, np.where(cols)[0][-1] + pad)
            
            crop_h = y_max - y_min
            crop_w = x_max - x_min
            
            # --- OTIMIZAÇÃO: Fator de Downsampling ---
            # Se a imagem for > 2000px, reduz para ~2000px para plotagem rápida
            target_size = 2000
            step = max(1, int(max(crop_h, crop_w) / target_size))
            
            crop_slice = (slice(y_min, y_max), slice(x_min, x_max))
            view_slice = (slice(y_min, y_max, step), slice(x_min, x_max, step))
        else:
            crop_slice = (slice(None), slice(None))
            view_slice = (slice(None, None, 5), slice(None, None, 5)) # Default agressivo se não achar ROI

        # 2. Configurações de Contraste (Baseado no GT cropado, mas decimado)
        gt_arr, _ = self.load_geotiff(gt_path)
        gt_view = gt_arr[view_slice] # Versão leve para visualização
        gt_crop = gt_arr[crop_slice] # Versão full para gráfico de perfil
        del gt_arr
        
        # Calcula vmin/vmax na versão leve
        valid_pixels = gt_view[np.isfinite(gt_view) & (gt_view > -1e30)]
        if valid_pixels.size > 0:
            vmin, vmax = np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)
        else:
            vmin, vmax = None, None
        
        # Plota Ground Truth (Leve)
        self._save_single_map(gt_view, "dtm_ground_truth.jpg", "Ground Truth (Real)", cmap="terrain", vmin=vmin, vmax=vmax)
        del gt_view, valid_pixels
        gc.collect()

        # 3. Orthophoto (Leve)
        ortho_arr, _ = self.load_geotiff(ortho_path)
        if ortho_arr is not None:
            if ortho_arr.ndim == 3:
                # Trata canais se for RGB
                ortho_view = ortho_arr[:, view_slice[0], view_slice[1]] if ortho_arr.shape[0] < ortho_arr.shape[1] else ortho_arr[view_slice[0], view_slice[1], :]
            else:
                ortho_view = ortho_arr[view_slice]
            
            del ortho_arr
            self._save_single_map(ortho_view, "orthophoto.jpg", "Ortoimagem", cmap="gray")
            del ortho_view
            gc.collect()

        # 4. Input NoData (Leve)
        inp_arr, _ = self.load_geotiff(input_path)
        if inp_arr is not None:
            inp_view = inp_arr[view_slice]
            del inp_arr
            self._save_single_map(inp_view, "dtm_nodata.jpg", "Input (Com Lacunas)", cmap="terrain", vmin=vmin, vmax=vmax)
            del inp_view
            gc.collect()

        # 5. Preenchido (Leve + Gráfico Full)
        filled_arr, _ = self.load_geotiff(filled_path)
        if filled_arr is not None:
            filled_view = filled_arr[view_slice]
            
            self._save_single_map(filled_view, "dtm_preenchido.jpg", "Resultado (Preenchido)", cmap="terrain", vmin=vmin, vmax=vmax)
            
            # Recarrega Máscara para Erro e Perfil
            mask_full, _ = self.load_geotiff(mask_path)
            mask_view = mask_full[view_slice]
            
            # 6. Mapa de Erro (Leve)
            # Precisamos de GT leve de novo para o diff
            gt_full_reload, _ = self.load_geotiff(gt_path)
            gt_view_reload = gt_full_reload[view_slice]
            del gt_full_reload

            diff = np.abs(gt_view_reload - filled_view)
            diff[mask_view == 0] = 0
            diff[gt_view_reload < -1e30] = 0
            
            self._save_single_map(diff, "error_map.jpg", f"Erro Absoluto (RMSE: {metrics.get('rmse_m', 0):.2f}m)", cmap="inferno")
            del diff, gt_view_reload, filled_view, mask_view

            # 7. Gráfico de Perfil (PRECISA DE ALTA RESOLUÇÃO)
            # Usamos as versões 'crop' (full res, apenas ROI)
            filled_crop = filled_arr[crop_slice]
            mask_crop = mask_full[crop_slice]
            
            self._save_profile_graph(gt_crop, filled_crop, mask_crop, "profile_graph.jpg")
            
            del filled_crop, gt_crop, mask_crop, filled_arr, mask_full
            gc.collect()

    def _save_single_map(self, data, filename, title, cmap="terrain", vmin=None, vmax=None):
        if data is None: return
        # Figura menor pois é só para JPEG
        fig = plt.figure(figsize=(8, 6))
        
        if cmap == "gray":
            valid_viz = data[data > 0]
            if valid_viz.size > 0:
                local_vmin, local_vmax = np.percentile(valid_viz, 2), np.percentile(valid_viz, 98)
                plt.imshow(data, cmap=cmap, vmin=local_vmin, vmax=local_vmax)
            else:
                plt.imshow(data, cmap=cmap)
        else:
            plot_data = np.where(data < -1e30, np.nan, data)
            plt.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.title(title, fontsize=12)
        if cmap != "gray":
            plt.colorbar(fraction=0.046, pad=0.04, label="m")
        plt.axis('off')
        
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches='tight', dpi=100) # DPI menor para velocidade
        plt.close(fig)

    def _save_profile_graph(self, gt, filled, mask, filename):
        # Gráfico precisa de boa resolução
        fig = plt.figure(figsize=(10, 5))
        
        mid_y = gt.shape[0] // 2
        line_gt = gt[mid_y, :]
        line_pred = filled[mid_y, :]
        x_axis = np.arange(len(line_gt))
        
        valid_mask = (line_gt > -10000) & (line_pred > -10000)
        
        if np.sum(valid_mask) == 0:
            rows_with_holes = np.where(np.any(mask > 0, axis=1))[0]
            if len(rows_with_holes) > 0:
                mid_y = rows_with_holes[len(rows_with_holes)//2]
                line_gt = gt[mid_y, :]
                line_pred = filled[mid_y, :]
                valid_mask = (line_gt > -10000) & (line_pred > -10000)

        x_plot = x_axis[valid_mask]
        gt_plot = line_gt[valid_mask]
        pred_plot = line_pred[valid_mask]
        
        if len(x_plot) > 0:
            plt.plot(x_plot, gt_plot, 'k-', lw=1.5, label='Real', alpha=0.8)
            plt.plot(x_plot, pred_plot, 'r--', lw=1.0, label='IA')

            hole_indices = np.where(mask[mid_y, :] > 0)[0]
            valid_hole_indices = np.intersect1d(hole_indices, x_plot)
            
            if len(valid_hole_indices) > 0:
                from itertools import groupby, count
                for _, g in groupby(valid_hole_indices, key=lambda n, c=count(): n-next(c)):
                    group = list(g)
                    plt.axvspan(group[0], group[-1], color='yellow', alpha=0.3)
                plt.axvspan(valid_hole_indices[0], valid_hole_indices[0], color='yellow', alpha=0.3, label='Lacuna')

        plt.title(f"Perfil (Y={mid_y})", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches='tight', dpi=120)
        plt.close(fig)

    def _calculate_masked_ssim(self, gt, pred, mask):
        valid_pixels = gt[np.isfinite(gt) & (gt > -1e30)]
        if valid_pixels.size == 0: return 0.0
        mn, mx = valid_pixels.min(), valid_pixels.max()
        scale = mx - mn + 1e-6
        
        gt_t = torch.tensor(gt, device=self.device).unsqueeze(0).unsqueeze(0)
        pred_t = torch.tensor(pred, device=self.device).unsqueeze(0).unsqueeze(0)
        mask_t = torch.tensor(mask, device=self.device).unsqueeze(0).unsqueeze(0)
        pred_masked = torch.where(mask_t, pred_t, gt_t)
        gt_norm = torch.clamp((gt_t - mn) / scale, 0, 1)
        pred_norm = torch.clamp((pred_masked - mn) / scale, 0, 1)
        
        score = self.ssim_calc(pred_norm, gt_norm).item()
        del gt_t, pred_t, mask_t, pred_masked, gt_norm, pred_norm
        return score

    def _return_empty_metrics(self):
        empty = {"rmse_m": 0.0, "mae_m": 0.0, "ssim": 0.0, "execution_time_s": 0.0, "evaluated_pixels": 0}
        self._save_metrics(empty)
        return empty

    def _save_metrics(self, metrics):
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
