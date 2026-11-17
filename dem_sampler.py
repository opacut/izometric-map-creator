import glob
import geo
import math
import os
from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from constants import REFERENCE_ELEVATION
from constants import TERRAIN_EXT_GRID_NODES
from constants import TERRAIN_EXT_RADIUS
from constants import TERRAIN_GRID_STEP

class DemSampler:
    def __init__(self, data_dir='data'):
        """Load Terrarium PNG tiles, remove SRTM seam pits, and prepare the bilinear sampler."""
        dem_files = glob.glob(f'{data_dir}/dem/*.png') or glob.glob('data/dem/*.png')
        tile_arrays = {}
        self.ext_heights = None
        
        for filepath in dem_files:
            _, tx_str, ty_str = os.path.basename(filepath)[:-4].split('_')
            rgb = np.asarray(Image.open(filepath).convert('RGB')).astype(np.float64)
            tile_arrays[(int(tx_str), int(ty_str))] = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768

        if not tile_arrays:
            self.mosaic = None
            self.sorted_xs = []
            self.sorted_ys = []
            return

        self.sorted_xs = sorted({coord[0] for coord in tile_arrays})
        self.sorted_ys = sorted({coord[1] for coord in tile_arrays})
        self.mosaic = np.zeros((len(self.sorted_ys) * 256, len(self.sorted_xs) * 256))

        for (tile_x, tile_y), tile_data in tile_arrays.items():
            r = (tile_y - self.sorted_ys[0]) * 256
            c = (tile_x - self.sorted_xs[0]) * 256
            self.mosaic[r:r + 256, c:c + 256] = tile_data

        # Fix seam pits with a 7x7 median filter
        padded = np.pad(self.mosaic, 3, mode='edge')
        local_median = np.median(sliding_window_view(padded, (7, 7)), axis=(2, 3))
        seam_pits = self.mosaic < local_median - 12
        self.mosaic = np.where(seam_pits, local_median, self.mosaic)
        
        if seam_pits.any():
            print(f'oprava terénu (pruhy ve zdrojových datech) {int(seam_pits.sum())} bodů')


    def recalculate_ext_heights(self):
        grid_xs, grid_ys = np.meshgrid(
            np.linspace(-TERRAIN_EXT_RADIUS, TERRAIN_EXT_RADIUS, TERRAIN_EXT_GRID_NODES),
            np.linspace(TERRAIN_EXT_RADIUS, -TERRAIN_EXT_RADIUS, TERRAIN_EXT_GRID_NODES),
        )
        #ext_heights = self.sample_elevation(grid_xs, grid_ys)
        self.ext_heights = sliding_window_view(np.pad(self.sample_elevation(grid_xs, grid_ys), 1, mode='edge'), (3, 3)).mean(axis=(2, 3)) - REFERENCE_ELEVATION


    def sample_elevation(self, local_x, local_y):
        if self.mosaic is None:
            return np.zeros_like(local_x)

        zoom_tiles = 2 ** 14
        lon, lat = geo.convert_local_meters_to_lonlat(local_x, local_y)
        grid_x = (lon + 180) / 360 * zoom_tiles * 256 - self.sorted_xs[0] * 256
        grid_y = (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * zoom_tiles * 256 - self.sorted_ys[0] * 256

        col0 = np.floor(grid_x - 0.5).astype(int)
        row0 = np.floor(grid_y - 0.5).astype(int)
        frac_x = grid_x - 0.5 - col0
        frac_y = grid_y - 0.5 - row0

        col0 = np.clip(col0, 0, self.mosaic.shape[1] - 2)
        row0 = np.clip(row0, 0, self.mosaic.shape[0] - 2)

        return (
            self.mosaic[row0, col0] * (1 - frac_x) * (1 - frac_y)
            + self.mosaic[row0, col0 + 1] * frac_x * (1 - frac_y)
            + self.mosaic[row0 + 1, col0] * (1 - frac_x) * frac_y
            + self.mosaic[row0 + 1, col0 + 1] * frac_x * frac_y
        )

    def get_ground_elevation(self, x, y):
        fx = (np.asarray(x) + TERRAIN_EXT_RADIUS) / TERRAIN_GRID_STEP
        fy = (TERRAIN_EXT_RADIUS - np.asarray(y)) / TERRAIN_GRID_STEP
        col0 = np.clip(np.floor(fx).astype(int), 0, TERRAIN_EXT_GRID_NODES - 2)
        row0 = np.clip(np.floor(fy).astype(int), 0, TERRAIN_EXT_GRID_NODES - 2)
        tx = np.clip(fx - col0, 0, 1)
        ty = np.clip(fy - row0, 0, 1)
        return (
            self.ext_heights[row0, col0] * (1 - tx) * (1 - ty)
            + self.ext_heights[row0, col0 + 1] * tx * (1 - ty)
            + self.ext_heights[row0 + 1, col0] * (1 - tx) * ty
            + self.ext_heights[row0 + 1, col0 + 1] * tx * ty
        )