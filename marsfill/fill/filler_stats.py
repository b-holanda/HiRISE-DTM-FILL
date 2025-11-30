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
            # Tenta ignorar erro se for opcional, mas loga aviso
            return None, None
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        
        if nodata is None:
            if np.nanmin(array) < -1e30:
                nodata = -3.4028235e38
            else:
                nodata = None
        return array, nodata

    def calculate_metrics(self, gt_path, filled_path, mask_path=None):
        logger.info("⚡ Calculando métricas...")
        start_time = time.time()

        gt_arr, gt_nodata = self.load_geotiff(gt_path)
        filled_arr, _ = self.load_geotiff(filled_path)

        if gt_arr is None or filled_arr is None:
            logger.error("Falha ao carregar arquivos para métricas.")
            return self._return_empty_metrics()

        # Definição da Máscara
        if mask_path:
            mask_arr, _ = self.load_geotiff(mask_path)
            # Crop para garantir dimensões iguais
            h = min(mask_arr.shape[0], gt_arr.shape[0])
            w = min(mask_arr.shape[1], gt_arr.shape[1])
            mask_arr = mask_arr[:h, :w]
            gt_arr = gt_arr[:h, :w]
            filled_arr = filled_arr[:h, :w]
            hole_mask = (mask_arr > 0)
        else:
            hole_mask = np.ones_like(gt_arr, dtype=bool)

        # Validação
        if np.sum(hole_mask) == 0:
            return self._return_empty_metrics()

        valid_gt_mask = np.isfinite(gt_arr) & (gt_arr > -1e30)
        if gt_nodata is not None:
             valid_gt_mask &= ~np.isclose(gt_arr, gt_nodata)

        eval_mask = hole_mask & valid_gt_mask
        
        num_pixels = int(np.sum(eval_mask))
        if num_pixels == 0:
            return self._return_empty_metrics()

        # Métricas
        y_true = gt_arr[eval_mask]
        y_pred = filled_arr[eval_mask]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
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

    def generate_all_outputs(self, gt_path, input_path, ortho_path, filled_path, mask_path, metrics):
        """
        Gera todas as imagens e gráficos solicitados separadamente.
        """
        # Carrega todos os arrays necessários
        gt_arr, _ = self.load_geotiff(gt_path)
        input_arr, _ = self.load_geotiff(input_path)
        ortho_arr, _ = self.load_geotiff(ortho_path)
        filled_arr, _ = self.load_geotiff(filled_path)
        mask_arr, _ = self.load_geotiff(mask_path)

        if gt_arr is None: return

        # Recorte Automático (Crop) baseado na máscara para focar na ação
        rows = np.any(mask_arr > 0, axis=1)
        cols = np.any(mask_arr > 0, axis=0)
        
        if np.any(rows):
            pad = 50
            y_min, y_max = max(0, np.where(rows)[0][0] - pad), min(gt_arr.shape[0], np.where(rows)[0][-1] + pad)
            x_min, x_max = max(0, np.where(cols)[0][0] - pad), min(gt_arr.shape[1], np.where(cols)[0][-1] + pad)
        else:
            y_min, y_max, x_min, x_max = 0, gt_arr.shape[0], 0, gt_arr.shape[1]

        # Função auxiliar de corte
        def crop(arr): 
            if arr is None: return None
            # Garante que arrays 2D e 3D (ortho) sejam cortados corretamente
            if arr.ndim == 3: return arr[:, y_min:y_max, x_min:x_max] if arr.shape[0] < arr.shape[1] else arr[y_min:y_max, x_min:x_max, :]
            return arr[y_min:y_max, x_min:x_max]

        gt_c = crop(gt_arr)
        inp_c = crop(input_arr)
        ortho_c = crop(ortho_arr)
        filled_c = crop(filled_arr)
        mask_c = crop(mask_arr)

        # Determina Escala de Cores Comum (Baseada no GT Válido)
        valid_pixels = gt_c[np.isfinite(gt_c) & (gt_c > -1e30)]
        if valid_pixels.size > 0:
            vmin, vmax = np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)
        else:
            vmin, vmax = None, None

        logger.info("🎨 Gerando imagens individuais...")

        # 1. Orthophoto
        self._save_single_map(ortho_c, "orthophoto.jpg", "Ortoimagem (Referência Visual)", cmap="gray")

        # 2. Ground Truth
        self._save_single_map(gt_c, "dtm_ground_truth.jpg", "Ground Truth (Real)", cmap="terrain", vmin=vmin, vmax=vmax)

        # 3. Input com NoData
        self._save_single_map(inp_c, "dtm_nodata.jpg", "Input (Com Lacunas)", cmap="terrain", vmin=vmin, vmax=vmax)

        # 4. Preenchido
        self._save_single_map(filled_c, "dtm_preenchido.jpg", "Resultado (Preenchido)", cmap="terrain", vmin=vmin, vmax=vmax)

        # 5. Mapa de Erro
        diff = np.abs(gt_c - filled_c)
        # Mascara onde não era buraco
        diff[mask_c == 0] = 0 
        diff[gt_c < -1e30] = 0
        self._save_single_map(diff, "error_map.jpg", f"Erro Absoluto (RMSE: {metrics.get('rmse_m', 0):.2f}m)", cmap="inferno")

        # 6. Gráfico de Perfil
        self._save_profile_graph(gt_c, filled_c, mask_c, "profile_graph.jpg")

    def _save_single_map(self, data, filename, title, cmap="terrain", vmin=None, vmax=None):
        if data is None: return
        
        plt.figure(figsize=(10, 8))
        
        # Tratamento para NoData na visualização
        if cmap != "gray": # DTMs
            plot_data = np.where(data < -1e30, np.nan, data)
        else: # Ortho
            plot_data = data

        plt.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(title, fontsize=14)
        plt.colorbar(fraction=0.046, pad=0.04, label="Metros" if cmap != "gray" and cmap != "inferno" else "")
        plt.axis('off')
        
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()

    def _save_profile_graph(self, gt, filled, mask, filename):
        plt.figure(figsize=(12, 6))
        
        mid_y = gt.shape[0] // 2
        line_gt = gt[mid_y, :]
        line_pred = filled[mid_y, :]
        x_axis = np.arange(len(line_gt))
        
        valid = line_gt > -1e30
        
        plt.plot(x_axis[valid], line_gt[valid], 'k-', lw=2.5, label='Ground Truth', alpha=0.8)
        plt.plot(x_axis[valid], line_pred[valid], 'r--', lw=2, label='Predição IA')

        # Destaque da área preenchida
        hole_indices = np.where(mask[mid_y, :] > 0)[0]
        if len(hole_indices) > 0:
            from itertools import groupby, count
            for _, g in groupby(hole_indices, key=lambda n, c=count(): n-next(c)):
                group = list(g)
                plt.axvspan(group[0], group[-1], color='yellow', alpha=0.3)
            plt.axvspan(0, 0, color='yellow', alpha=0.3, label='Lacuna') # Legenda fake

        plt.title("Perfil Topográfico (Transecto Central)", fontsize=14)
        plt.xlabel("Pixels")
        plt.ylabel("Elevação (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()

    # -- Métodos Auxiliares Mantidos --
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
        return self.ssim_calc(pred_norm, gt_norm).item()

    def _return_empty_metrics(self):
        empty = {"rmse_m": 0.0, "mae_m": 0.0, "ssim": 0.0, "execution_time_s": 0.0, "evaluated_pixels": 0}
        self._save_metrics(empty)
        return empty, None, None, None

    def _save_metrics(self, metrics):
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
