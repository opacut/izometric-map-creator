"""Preprocessing: Overture Maps + Terrarium DEM -> build/scene.json, build/ground_<season>.png, build/lightmap.png, build/info_src.json

Can be run as a script or imported and called as a function.
"""
import glob
import hashlib
import json
import math
import os
import random
import sys

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import box
from numpy.lib.stride_tricks import sliding_window_view
from shapely.ops import transform
from constants import CHUNK_MARGIN_METERS
from constants import TERRAIN_FLOOR_ELEVATION
from constants import HALF_SQUARE_SIDE
from constants import MAP_CHUNKS
from constants import REFERENCE_ELEVATION

import geo


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def hash_to_unit_float(value):
    """Deterministic 0..1 float from a string or value."""
    digest = hashlib.md5(str(value).encode()).hexdigest()[:8]
    return int(digest, 16) / 0xffffffff


def world_coord_key(local_x, local_y, center_x, center_y):
    """World-coordinate key consistent across adjacent map chunks."""
    return f'{round(local_x + center_x, 1)},{round(local_y + center_y, 1)}'


def vectorized_int_hash(*arrays):
    """Vectorized deterministic numpy hash returning values in 0..1."""
    with np.errstate(over='ignore'):
        state = np.uint64(0x9E3779B97F4A7C15)
        for arr in arrays:
            arr_u64 = np.asarray(arr).astype(np.int64).astype(np.uint64)
            state = state ^ (
                arr_u64 * np.uint64(0xBF58476D1CE4E5B9)
                + np.uint64(0x94D049BB133111EB)
                + (state << np.uint64(6))
                + (state >> np.uint64(2))
            )
        state = (state ^ (state >> np.uint64(31))) * np.uint64(0xBF58476D1CE4E5B9)
        state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        state = state ^ (state >> np.uint64(31))
    return (state >> np.uint64(11)).astype(np.float64) / float(2 ** 53)


def extract_polygons(geometry):
    """Extract valid Polygons from any Shapely geometry."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if hasattr(geometry, 'geoms'):
        return [poly for sub_geom in geometry.geoms for poly in extract_polygons(sub_geom)]
    return []


def extract_linestrings(geometry):
    """Extract valid LineStrings from any Shapely geometry."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if hasattr(geometry, 'geoms'):
        return [line for sub_geom in geometry.geoms for line in extract_linestrings(sub_geom)]
    return []


def get_primary_name(record):
    """Extract primary name string from a feature record."""
    names_dict = record.get('names')
    return names_dict.get('primary') if isinstance(names_dict, dict) else None


def load_spatial_parquet(data_dir, layer_name):
    """Load a Parquet spatial layer and reproject WKB geometries to local chunk coordinates."""
    df = pq.read_table(f'{data_dir}/{layer_name}.parquet').to_pandas()
    df = df.sort_values('id', kind='stable').reset_index(drop=True)
    df['geom'] = df.geometry.map(
        lambda wkb: transform(lambda lon, lat, z=None: geo.convert_lonlat_to_local_meters(lon, lat), shapely.from_wkb(wkb))
    )
    return df


# ==========================================
# TERRAIN & DEM SAMPLER
# ==========================================

def build_dem_sampler(data_dir):
    """Load Terrarium PNG tiles, remove SRTM seam pits, and return a bilinear elevation sampler."""
    tile_arrays = {}
    dem_files = glob.glob(f'{data_dir}/dem/*.png') or glob.glob('data/dem/*.png')
    for filepath in dem_files:
        _, tile_x_str, tile_y_str = os.path.basename(filepath)[:-4].split('_')
        rgb = np.asarray(Image.open(filepath).convert('RGB')).astype(np.float64)
        elevation_m = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
        tile_arrays[(int(tile_x_str), int(tile_y_str))] = elevation_m

    if not tile_arrays:
        return lambda px, py: np.zeros_like(px)

    sorted_xs = sorted({coord[0] for coord in tile_arrays})
    sorted_ys = sorted({coord[1] for coord in tile_arrays})
    mosaic = np.zeros((len(sorted_ys) * 256, len(sorted_xs) * 256))

    for (tile_x, tile_y), tile_data in tile_arrays.items():
        row_start = (tile_y - sorted_ys[0]) * 256
        col_start = (tile_x - sorted_xs[0]) * 256
        mosaic[row_start:row_start + 256, col_start:col_start + 256] = tile_data

    # Fix narrow seam pits in Terrarium SRTM tiles using a 7x7 median filter
    padded_mosaic = np.pad(mosaic, 3, mode='edge')
    local_median = np.median(sliding_window_view(padded_mosaic, (7, 7)), axis=(2, 3))
    seam_pits = mosaic < local_median - 12
    mosaic = np.where(seam_pits, local_median, mosaic)
    if seam_pits.any():
        print('oprava terénu (pruhy ve zdrojových datech)', int(seam_pits.sum()), 'bodů')

    zoom_tiles = 2 ** 14

    def sample_elevation(local_x, local_y):
        lon, lat = geo.convert_local_meters_to_lonlat(local_x, local_y)
        grid_x = (lon + 180) / 360 * zoom_tiles * 256 - sorted_xs[0] * 256
        grid_y = (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * zoom_tiles * 256 - sorted_ys[0] * 256

        col0 = np.floor(grid_x - 0.5).astype(int)
        row0 = np.floor(grid_y - 0.5).astype(int)
        frac_x = grid_x - 0.5 - col0
        frac_y = grid_y - 0.5 - row0

        col0 = np.clip(col0, 0, mosaic.shape[1] - 2)
        row0 = np.clip(row0, 0, mosaic.shape[0] - 2)

        return (
            mosaic[row0, col0] * (1 - frac_x) * (1 - frac_y)
            + mosaic[row0, col0 + 1] * frac_x * (1 - frac_y)
            + mosaic[row0 + 1, col0] * (1 - frac_x) * frac_y
            + mosaic[row0 + 1, col0 + 1] * frac_x * frac_y
        )

    return sample_elevation


# ==========================================
# MAIN PREPARATION PIPELINE
# ==========================================

def prepare_scene(data_dir='data', chunk_coord=(0, 0)):
    """Preprocess Overture Maps & Terrarium DEM into scene geometry, textures, and metadata."""
    data_dir="data_all"
    chunk = (
        tuple(int(val) for val in chunk_coord.split(','))
        if isinstance(chunk_coord, str)
        else chunk_coord
    )

    out_dir = 'build' if tuple(chunk) == (0, 0) else f'build/c{chunk[0]}_{chunk[1]}'
    center_x = chunk[0] * 2 * HALF_SQUARE_SIDE
    center_y = chunk[1] * 2 * HALF_SQUARE_SIDE

    os.makedirs(out_dir, exist_ok=True)

    # Spatial bounds & margin definitions
    tex_meters_per_px = 2 * HALF_SQUARE_SIDE / 8000  # Max GPU texture size is 8192 px
    chunk_box = box(-HALF_SQUARE_SIDE, -HALF_SQUARE_SIDE, HALF_SQUARE_SIDE, HALF_SQUARE_SIDE)
    # Returns neighboring chunks (including diagonals) that exist in the map: [(delta_east, delta_north), ...]
    neighbor_offsets = [
        (delta_east, delta_north) 
        for delta_north in (1, 0, -1) 
        for delta_east in (-1, 0, 1) 
        if (delta_east or delta_north) and (chunk[0] + delta_east, chunk[1] + delta_north) in MAP_CHUNKS
    ]
    extended_half = HALF_SQUARE_SIDE + CHUNK_MARGIN_METERS
    margin_px = round(CHUNK_MARGIN_METERS / tex_meters_per_px)
    chunk_size = 2 * HALF_SQUARE_SIDE

    neighbor_boxes = [
        box(dx * chunk_size - HALF_SQUARE_SIDE, dy * chunk_size - HALF_SQUARE_SIDE, dx * chunk_size + HALF_SQUARE_SIDE, dy * chunk_size + HALF_SQUARE_SIDE)
        for dx, dy in neighbor_offsets
    ]
    map_clip_area = shapely.unary_union([chunk_box] + neighbor_boxes)
    extended_area = map_clip_area.intersection(box(-extended_half, -extended_half, extended_half, extended_half))
    extended_area_prep = shapely.prepared.prep(extended_area)

    satellite_data_path_name = 'asset/satelit.json'
    manual_overrides_path_name = 'assets/upresneni.json'
    strechy_orto_path_name = 'assets/strechy_orto.json'
    custom_prep_path_name = 'assets/custom_prep.py'
    ruian_buildings_path_name = 'assets/ruian_budovy.json'
    area_overrides_path_name = 'assets/oblasti.json'
    roof_ridges_path_name = 'assets/hrebeny.json'
    lidar_heights_path_name = 'assets/dmp_vysky.json'
    manual_lidar_path_name = 'assets/dmp_rucne.json'

    if os.path.exists(satellite_data_path_name):
        satellite_data = json.load(open(satellite_data_path_name))
    else:
        satellite_data = {'b': {}, 'lu': {}, 'zones': []}

    if os.path.exists(manual_overrides_path_name):
        manual_overrides = json.load(open(manual_overrides_path_name))
    else:
        manual_overrides = {'b': {}, 'roofs': [], 'paved': [], 'buses': []}

    if os.path.exists(strechy_orto_path_name):
        for bld_id, roof_color in json.load(open(strechy_orto_path_name)).items():
            satellite_data['b'].setdefault(bld_id, {})['rc'] = roof_color

    for bld_id, props in manual_overrides['b'].items():
        satellite_data['b'][bld_id] = {**satellite_data['b'].get(bld_id, {}), **props}

    for lu_id, surface_type in manual_overrides.get('lu', {}).items():
        satellite_data['lu'][lu_id] = surface_type

    if os.path.exists(custom_prep_path_name):
        custom_hook_code = open(custom_prep_path_name, encoding='utf-8').read()
    else:
        custom_hook_code = None

    if os.path.exists(ruian_buildings_path_name):
        ruian_buildings = json.load(open(ruian_buildings_path_name))
    else:
        ruian_buildings = {}

    if os.path.exists(area_overrides_path_name):
        area_overrides = json.load(open(area_overrides_path_name))
    else:
        area_overrides = {}

    for area_cfg in area_overrides.values():
        combined_buildings = list(area_cfg['b_auto'].items()) + list(area_cfg['b'].items())
        for bld_id, props in combined_buildings:
            satellite_data['b'][bld_id] = {**satellite_data['b'].get(bld_id, {}), **props}

    if os.path.exists(roof_ridges_path_name):
        roof_ridges = json.load(open(roof_ridges_path_name))
    else:
        roof_ridges = {}

    if (os.path.exists(lidar_heights_path_name)):
        lidar_heights = json.load(open(lidar_heights_path_name))
    else:
        lidar_heights = {}

    if os.path.exists(manual_lidar_path_name):
        manual_lidar = {
            k: v for k, v in json.load(open(manual_lidar_path_name)).items() if not k.startswith('_')
        }
        lidar_heights.update(manual_lidar)

    lidar_heights.update(manual_overrides.get('dmp', {}))


    def run_custom_hook(phase):
        if custom_hook_code:
            local_scope = locals()
            exec(custom_hook_code, globals(), local_scope)

    # ---------------- Terrain Grid ----------------
    print("Terrain Grid")
    dem_sampler = build_dem_sampler(data_dir)
    grid_step = 5.0
    grid_nodes = int(2 * HALF_SQUARE_SIDE / grid_step) + 1
    margin_steps = int(math.ceil(CHUNK_MARGIN_METERS / grid_step)) + 2
    ext_grid_nodes = grid_nodes + 2 * margin_steps
    ext_radius = HALF_SQUARE_SIDE + margin_steps * grid_step

    grid_xs, grid_ys = np.meshgrid(
        np.linspace(-ext_radius, ext_radius, ext_grid_nodes),
        np.linspace(ext_radius, -ext_radius, ext_grid_nodes),
    )
    ext_heights = dem_sampler(grid_xs, grid_ys)

    run_custom_hook('terrain')
    ext_heights = sliding_window_view(np.pad(ext_heights, 1, mode='edge'), (3, 3)).mean(axis=(2, 3))
    base_elevation = REFERENCE_ELEVATION
    ext_heights = ext_heights - base_elevation
    core_heights = ext_heights[margin_steps:margin_steps + grid_nodes, margin_steps:margin_steps + grid_nodes]

    def get_ground_elevation(x, y):
        fx = (np.asarray(x) + ext_radius) / grid_step
        fy = (ext_radius - np.asarray(y)) / grid_step
        col0 = np.clip(np.floor(fx).astype(int), 0, ext_grid_nodes - 2)
        row0 = np.clip(np.floor(fy).astype(int), 0, ext_grid_nodes - 2)
        tx = np.clip(fx - col0, 0, 1)
        ty = np.clip(fy - row0, 0, 1)
        return (
            ext_heights[row0, col0] * (1 - tx) * (1 - ty)
            + ext_heights[row0, col0 + 1] * tx * (1 - ty)
            + ext_heights[row0 + 1, col0] * (1 - tx) * ty
            + ext_heights[row0 + 1, col0 + 1] * tx * ty
        )

    print('terén', grid_nodes, 'x', grid_nodes, 'převýšení', round(float(core_heights.max()), 1), 'm')

    # ---------------- Surface Class Rasterization ----------------
    print("Surface Class Rasterization")
    SURFACE_IDS = dict(
        grass=0, garden=1, park=2, meadow=3, forest=4, scrub=5, pitch=6, track=7, sand=8, paved=9,
        industrial=10, parking=11, road=12, marking=13, curb=14, dirt=15, water=16, pool=17, stream=18,
        rail=19, cemetery=20, wetland=21, rock=22, construction=23, footway=24, pool_edge=25, farmland=26,
        cobble=27, grave=28, manhole=29, court_blue=30, zone=31, zone2=32, piste=33, skijump=34,
        wheat=35, wheat2=36, rape=37, rape2=38, maize=39, maize2=40, soil=41, soil2=42,
        plaza=43, plaza2=44, deck=45,
    )

    core_tex_width = int(2 * HALF_SQUARE_SIDE / tex_meters_per_px)
    ext_tex_width = core_tex_width + 2 * margin_px

    surface_img = Image.new('L', (ext_tex_width, ext_tex_width), SURFACE_IDS['grass'])
    surface_draw = ImageDraw.Draw(surface_img)
    tree_block_img = Image.new('L', (ext_tex_width, ext_tex_width), 0)
    tree_block_draw = ImageDraw.Draw(tree_block_img)
    draw_bounds = box(-extended_half, -extended_half, extended_half, extended_half).buffer(50)

    def to_tex_coords(coords):
        return [((x + extended_half) / tex_meters_per_px, (extended_half - y) / tex_meters_per_px) for x, y in coords]

    def to_tex_pixel(x, y):
        return int((x + extended_half) / tex_meters_per_px), int((extended_half - y) / tex_meters_per_px)

    def paint_polygon(geometry, class_id, outline_id=None, block_trees=False):
        for poly in extract_polygons(geometry.intersection(draw_bounds)):
            surface_draw.polygon(to_tex_coords(poly.exterior.coords), fill=class_id, outline=outline_id)
            for hole in poly.interiors:
                surface_draw.polygon(to_tex_coords(hole.coords), fill=SURFACE_IDS['grass'])
            if block_trees:
                tree_block_draw.polygon(to_tex_coords(poly.exterior.coords), fill=255)

    def paint_line(geometry, class_id, width_m, block_trees=False):
        for line in extract_linestrings(geometry.intersection(draw_bounds)):
            pts = to_tex_coords(line.coords)
            px_width = max(1, round(width_m / tex_meters_per_px))
            surface_draw.line(pts, fill=class_id, width=px_width, joint='curve')
            if block_trees:
                block_px_width = max(1, round((width_m + 2) / tex_meters_per_px))
                tree_block_draw.line(pts, fill=255, width=block_px_width, joint='curve')

    # ---------------- Load Spatial Layers ----------------
    land_use_df = load_spatial_parquet(data_dir, 'land_use')
    land_use_df['area'] = land_use_df.geom.map(lambda g: g.area)
    land_use_df = land_use_df.sort_values('area', ascending=False)

    land_cover_df = load_spatial_parquet(data_dir, 'land')
    infra_df = load_spatial_parquet(data_dir, 'infrastructure')
    water_df = load_spatial_parquet(data_dir, 'water')
    segments_df = load_spatial_parquet(data_dir, 'segment')
    buildings_df = load_spatial_parquet(data_dir, 'building')

    # ---------------- Land Use & Vegetation Masks ----------------
    print("Land Use & Vegetation Masks")
    LAND_USE_MAP = {
        'residential': 'garden', 'garden': 'garden', 'grass': 'park', 'park': 'park', 'village_green': 'park',
        'recreation_ground': 'park', 'meadow': 'meadow', 'farmland': 'farmland', 'allotments': 'garden',
        'cemetery': 'cemetery', 'pitch': 'pitch', 'track': 'track', 'playground': 'sand', 'stadium': 'pitch',
        'school': 'paved', 'kindergarten': 'garden', 'industrial': 'industrial', 'garages': 'industrial',
        'commercial': 'paved', 'retail': 'paved', 'pedestrian': 'cobble', 'plaza': 'cobble',
        'construction': 'construction', 'railway': 'dirt', 'barracks': 'industrial',
        'downhill': 'piste', 'ski_jump': 'skijump',
    }
    NO_TREE_LAND_USES = {
        'pitch', 'track', 'playground', 'stadium', 'school', 'industrial', 'garages', 'commercial',
        'retail', 'pedestrian', 'plaza', 'construction', 'railway', 'downhill', 'ski_jump',
    }

    green_mask_data = None
    if os.path.exists('build/zelen_maska.png'):
        green_arr = np.asarray(Image.open('build/zelen_maska.png')) > 0
        if os.path.exists('build/les_maska.png'):
            forest_arr = np.asarray(Image.open('build/les_maska.png')) > 0
            if forest_arr.shape == green_arr.shape:
                green_arr = green_arr | forest_arr
        green_mask_data = (green_arr, json.load(open('build/zelen_maska.json'))['x0y1m'])

    def compute_green_share(geometry):
        if green_mask_data is None:
            return 0.0
        minx, miny, maxx, maxy = geometry.bounds
        xs, ys = np.meshgrid(np.arange(minx, maxx, 5.0), np.arange(miny, maxy, 5.0))
        inside = shapely.contains_xy(geometry, xs, ys)
        if inside.sum() < 8:
            return 0.0
        mask_arr, (origin_x, origin_y, res) = green_mask_data
        rows = np.clip(((origin_y - (ys[inside] + center_y)) / res).astype(int), 0, mask_arr.shape[0] - 1)
        cols = np.clip(((xs[inside] + center_x - origin_x) / res).astype(int), 0, mask_arr.shape[1] - 1)
        return float(mask_arr[rows, cols].mean())

    green_plaza_ids = set()
    for _, row in land_use_df.iterrows():
        surf_name = satellite_data['lu'].get(row.id) or LAND_USE_MAP.get(row['class'])
        should_block_trees = row['class'] in NO_TREE_LAND_USES
        if surf_name == 'cobble' and compute_green_share(row.geom) > 0.5:
            surf_name = 'park'
            should_block_trees = False
            green_plaza_ids.add(row.id)
        if surf_name:
            paint_polygon(row.geom, SURFACE_IDS[surf_name], block_trees=should_block_trees)

    if os.path.exists('build/zelen_maska.png'):
        green_arr = np.asarray(Image.open('build/zelen_maska.png')) > 0
        origin_x, origin_y, res = json.load(open('build/zelen_maska.json'))['x0y1m']
        pixel_centers = (np.arange(ext_tex_width) + 0.5) * tex_meters_per_px - extended_half
        cols = np.clip(((pixel_centers + center_x - origin_x) / res).astype(int), 0, green_arr.shape[1] - 1)
        rows = np.clip(((origin_y - (-pixel_centers + center_y)) / res).astype(int), 0, green_arr.shape[0] - 1)

        surf_arr = np.array(surface_img)
        reclaimed_green = green_arr[np.ix_(rows, cols)] & np.isin(
            surf_arr, [SURFACE_IDS['paved'], SURFACE_IDS['industrial']]
        )
        surf_arr[reclaimed_green] = SURFACE_IDS['grass']
        surface_img.paste(Image.fromarray(surf_arr))

        block_arr = np.array(tree_block_img)
        block_arr[reclaimed_green] = 0
        tree_block_img.paste(Image.fromarray(block_arr))

    # ---------------- Sports Pitch Markings ----------------
    print("Sports Pitch Markings")
    pitch_polygons = [
        poly
        for _, row in land_use_df.iterrows()
        if row['class'] == 'pitch' and satellite_data['lu'].get(row.id, 'pitch') in ('pitch', 'track', 'court_blue')
        for poly in extract_polygons(row.geom.intersection(draw_bounds))
        if poly.area > 30
    ]
    merged_pitches = extract_polygons(shapely.unary_union(pitch_polygons)) if pitch_polygons else []
    for pitch in merged_pitches:
        obb = pitch.minimum_rotated_rectangle
        if pitch.area / max(obb.area, 1) < 0.86:
            continue
        inset = obb.buffer(-0.6, join_style=2)
        if inset.is_empty:
            continue
        corners = [np.array(c) for c in list(inset.exterior.coords)[:4]]
        edge_a, edge_b = corners[1] - corners[0], corners[3] - corners[0]
        if np.linalg.norm(edge_a) < np.linalg.norm(edge_b):
            corners = [corners[0], corners[3], corners[2], corners[1]]
            edge_a, edge_b = edge_b, edge_a
        length_m, width_m = np.linalg.norm(edge_a), np.linalg.norm(edge_b)
        unit_u, unit_v = edge_a / length_m, edge_b / width_m
        if length_m < 12 or width_m < 6:
            continue

        def pitch_pt(s, t):
            return tuple(corners[0] + unit_u * s + unit_v * t)

        surface_draw.polygon(
            to_tex_coords([pitch_pt(0, 0), pitch_pt(length_m, 0), pitch_pt(length_m, width_m), pitch_pt(0, width_m)]),
            outline=SURFACE_IDS['marking'],
        )
        surface_draw.line(
            to_tex_coords([pitch_pt(length_m / 2, 0), pitch_pt(length_m / 2, width_m)]),
            fill=SURFACE_IDS['marking'],
            width=1,
        )
        if length_m > 70 and width_m > 40:
            radius = 9.15
            mid_x, mid_y = pitch_pt(length_m / 2, width_m / 2)
            circle_pts = [
                (
                    mid_x + radius * math.cos(k / 24 * math.tau) * unit_u[0] + radius * math.sin(k / 24 * math.tau) * unit_v[0],
                    mid_y + radius * math.cos(k / 24 * math.tau) * unit_u[1] + radius * math.sin(k / 24 * math.tau) * unit_v[1],
                )
                for k in range(25)
            ]
            surface_draw.line(to_tex_coords(circle_pts), fill=SURFACE_IDS['marking'], width=1)
            for s0, s1 in ((0, 16.5), (length_m, length_m - 16.5)):
                w0, w1 = width_m / 2 - 20.15, width_m / 2 + 20.15
                surface_draw.line(
                    to_tex_coords([pitch_pt(s0, w0), pitch_pt(s1, w0), pitch_pt(s1, w1), pitch_pt(s0, w1)]),
                    fill=SURFACE_IDS['marking'],
                    width=1,
                )

    # ---------------- Natural Land Cover & Ski Pistes ----------------
    print("Natural Land Cover & Ski Pistes")
    LAND_COVER_MAP = {
        'forest': 'forest', 'wood': 'forest', 'scrub': 'scrub', 'shrub': 'scrub',
        'grassland': 'meadow', 'wetland': 'wetland', 'bare_rock': 'rock', 'rock': 'rock',
    }
    for _, row in land_cover_df.iterrows():
        if row['class'] in LAND_COVER_MAP and row.geom.geom_type != 'Point':
            paint_polygon(row.geom, SURFACE_IDS[LAND_COVER_MAP[row['class']]])

    ski_pistes = [line for _, row in land_use_df.iterrows() if row['class'] == 'downhill' for line in extract_linestrings(row.geom)]
    for piste_line in ski_pistes:
        paint_polygon(piste_line.buffer(22), SURFACE_IDS['piste'], block_trees=True)

    forest_mask = None
    if os.environ.get('IZO_LES', '1') != '0' and os.path.exists('build/les_maska.png'):
        Image.MAX_IMAGE_PIXELS = None
        mask_x0, mask_y1, mask_res = json.load(open('build/les_maska.json'))['x0y1m']
        crop_box = (
            round((center_x - extended_half - mask_x0) / mask_res),
            round((mask_y1 - center_y - extended_half) / mask_res),
            round((center_x + extended_half - mask_x0) / mask_res),
            round((mask_y1 - center_y + extended_half) / mask_res),
        )
        cropped_forest = Image.open('build/les_maska.png').crop(crop_box).resize((ext_tex_width, ext_tex_width), Image.NEAREST)
        forest_mask = np.asarray(cropped_forest) > 0
        non_forest_mask = ((np.asarray(surface_img) == SURFACE_IDS['forest']) & ~forest_mask).astype(np.uint8) * 255
        surface_img.paste(SURFACE_IDS['meadow'], mask=Image.fromarray(non_forest_mask))

    # ---------------- ZABAGED Agricultural Fields ----------------
    print("ZABAGED Agricultural Fields")
    def build_zabaged_polygon(rings):
        local_rings = [
            list(zip(*[np.asarray(v, float) for v in geo.convert_lonlat_to_local_meters([pt[1] for pt in ring], [pt[0] for pt in ring])]))
            for ring in rings
        ]
        return shapely.make_valid(Polygon(local_rings[0], local_rings[1:]))

    zabaged_data = (
        json.load(open('assets/zabaged.json'))
        if os.path.exists('assets/zabaged.json')
        else {'orna': [], 'ttp': [], 'sad': []}
    )
    ext_clip_box = box(-extended_half, -extended_half, extended_half, extended_half)
    arable_overrides = set(zabaged_data.get('pole', []))

    meadow_polys = [
        poly
        for idx, ring in enumerate(zabaged_data['ttp'])
        if idx not in arable_overrides
        for poly in [build_zabaged_polygon(ring)]
        if poly.intersects(ext_clip_box)
    ]
    arable_rings = zabaged_data['orna'] + [zabaged_data['ttp'][idx] for idx in sorted(arable_overrides)]
    arable_polys = [poly for ring in arable_rings for poly in [build_zabaged_polygon(ring)] if poly.intersects(ext_clip_box)]
    orchard_polys = [poly for poly in map(build_zabaged_polygon, zabaged_data['sad']) if poly.intersects(ext_clip_box)]

    for meadow_poly in meadow_polys:
        paint_polygon(meadow_poly, SURFACE_IDS['meadow'])

    CROP_TYPES = ['wheat'] * 4 + ['rape'] * 2 + ['maize'] * 2 + ['soil'] * 2
    for field_poly in arable_polys:
        obb = field_poly.minimum_rotated_rectangle
        corners = list(obb.exterior.coords)[:4]
        edge1 = np.subtract(corners[1], corners[0])
        edge2 = np.subtract(corners[2], corners[1])
        if np.linalg.norm(edge1) < np.linalg.norm(edge2):
            edge1, edge2 = edge2, edge1
        furrow_dir = edge1 / np.linalg.norm(edge1)
        if furrow_dir[0] < 0 or (furrow_dir[0] == 0 and furrow_dir[1] < 0):
            furrow_dir = -furrow_dir
        perp_dir = np.array([-furrow_dir[1], furrow_dir[0]])
        centroid_pt = np.array(field_poly.centroid.coords[0])
        crop_name = CROP_TYPES[
            int(hash_to_unit_float(world_coord_key(centroid_pt[0], centroid_pt[1], center_x, center_y) + 'pole') * len(CROP_TYPES))
        ]
        for sub_poly in extract_polygons(field_poly.intersection(draw_bounds)):
            if sub_poly.area < 400:
                continue
            paint_polygon(sub_poly, SURFACE_IDS[crop_name])
            rel_coords = np.array(sub_poly.exterior.coords) - centroid_pt
            proj_u, proj_v = rel_coords @ furrow_dir, rel_coords @ perp_dir
            offset_v = math.floor(proj_v.min() / 2.0) * 2.0
            while offset_v < proj_v.max():
                furrow_line = LineString([
                    tuple(centroid_pt + perp_dir * offset_v + furrow_dir * (proj_u.min() - 1)),
                    tuple(centroid_pt + perp_dir * offset_v + furrow_dir * (proj_u.max() + 1)),
                ]).intersection(sub_poly)
                if not furrow_line.is_empty:
                    paint_line(furrow_line, SURFACE_IDS[crop_name + '2'], 0.6)
                offset_v += 2.0

    # ---------------- Parking, Paved Areas & Water ----------------
    print("Parking, Paved Areas & Water")
    parking_polygons = []
    for _, row in infra_df.iterrows():
        if row['class'] == 'parking' and row.geom.geom_type in ('Polygon', 'MultiPolygon'):
            paint_polygon(row.geom, SURFACE_IDS['parking'], block_trees=True)
            parking_polygons += [(row.id, poly) for poly in extract_polygons(row.geom.intersection(chunk_box))]
        elif row['class'] == 'bus_station' and row.geom.geom_type in ('Polygon', 'MultiPolygon'):
            paint_polygon(row.geom, SURFACE_IDS['parking'], block_trees=True)
        elif (
            row['class'] == 'platform'
            and row.subtype == 'transit'
            and row.geom.geom_type in ('Polygon', 'MultiPolygon')
            and row.geom.area < 2000
        ):
            paint_polygon(row.geom, SURFACE_IDS['paved'], block_trees=True)

    for paved_coords in manual_overrides['paved']:
        paint_polygon(Polygon([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in paved_coords]), SURFACE_IDS['paved'], block_trees=True)

    render_streams = bool(geo.CITY_CONFIG.get('potoky'))
    building_stream_excl = (
        shapely.unary_union([poly.buffer(1.2) for geom in buildings_df.geom for poly in extract_polygons(geom) if poly.intersects(draw_bounds)])
        if render_streams
        else shapely.geometry.GeometryCollection()
    )
    for _, row in water_df.iterrows():
        geom = row.geom
        if geom.geom_type in ('Polygon', 'MultiPolygon'):
            if row['class'] == 'swimming_pool':
                paint_polygon(geom, SURFACE_IDS['pool'], outline_id=SURFACE_IDS['pool_edge'], block_trees=True)
            else:
                paint_polygon(geom, SURFACE_IDS['water'], block_trees=True)
        elif row['class'] == 'river':
            paint_line(geom, SURFACE_IDS['stream'], 6, block_trees=True)
        elif row['class'] == 'stream' and render_streams:
            visible_stream = geom.difference(building_stream_excl) if not building_stream_excl.is_empty else geom
            if not visible_stream.is_empty:
                stream_width = 4.8 if get_primary_name(row) else 2.8
                paint_line(visible_stream, SURFACE_IDS['stream'], stream_width, block_trees=True)

    # ---------------- Road Network, Tunnels & Bridges ----------------
    print("Road Network, Tunnels & Bridges")
    def is_under_construction(row):
        if row['class'] != 'motorway' or row.road_flags is None:
            return False
        return any('is_under_construction' in list(flag.get('values')) for flag in row.road_flags)

    segments_df = segments_df[[not is_under_construction(row) for _, row in segments_df.iterrows()]].reset_index(drop=True)
    MIN_TUNNEL_LEN = 60.0

    def get_segment_flags(row):
        flags = row.get('rail_flags') if row.subtype == 'rail' else row.road_flags
        return list(flags) if isinstance(flags, (list, np.ndarray)) else []

    def split_tunnel_intervals(row):
        flags = get_segment_flags(row)
        tunnels = [f.get('between') for f in flags if 'is_tunnel' in list(f.get('values'))]
        tunnels = [t for t in tunnels if (1.0 if t is None else float(t[1]) - float(t[0])) * row.geom.length >= MIN_TUNNEL_LEN]
        if not tunnels:
            return None
        if any(t is None for t in tunnels):
            return []
        visible_parts, cursor = [], 0.0
        for start_f, end_f in sorted((float(t[0]), float(t[1])) for t in tunnels):
            if start_f > cursor:
                visible_parts.append((cursor, start_f))
            cursor = max(cursor, end_f)
        if cursor < 1:
            visible_parts.append((cursor, 1.0))
        return visible_parts

    surface_segments, tunnel_portals = [], []
    for _, row in segments_df.iterrows():
        visible_intervals = split_tunnel_intervals(row)
        if visible_intervals is None:
            surface_segments.append(row)
            continue
        for start_f, end_f in visible_intervals:
            sub_row = row.copy()
            sub_row['geom'] = shapely.ops.substring(row.geom, start_f, end_f, normalized=True)
            surface_segments.append(sub_row)
        if row['class'] == 'trunk':
            for start_f, end_f in visible_intervals:
                for frac, direction in ((start_f, 1), (end_f, -1)):
                    if 0 < frac < 1:
                        pt0 = row.geom.interpolate(frac, normalized=True)
                        pt1 = row.geom.interpolate(max(0, min(1, frac + direction * 0.004)), normalized=True)
                        tunnel_portals.append((pt0.x, pt0.y, math.atan2(pt1.y - pt0.y, pt1.x - pt0.x)))
    segments_df = pd.DataFrame(surface_segments).reset_index(drop=True)

    MIN_BRIDGE_LEN = manual_overrides.get('bridge_min', 8.0)

    def split_bridge_intervals(row):
        flags = get_segment_flags(row)
        bridges = [f.get('between') for f in flags if 'is_bridge' in list(f.get('values'))]
        if not bridges:
            return None
        intervals = sorted((0.0, 1.0) if t is None else (float(t[0]), float(t[1])) for t in bridges)
        parts, cursor = [], 0.0
        for start_f, end_f in intervals:
            if start_f > cursor:
                parts.append((cursor, start_f, False))
            parts.append((max(cursor, start_f), end_f, True))
            cursor = max(cursor, end_f)
        if cursor < 1:
            parts.append((cursor, 1.0, False))
        return parts

    ground_segments, bridge_segments, raw_bridge_records = [], [], []
    for _, row in segments_df.iterrows():
        bridge_intervals = split_bridge_intervals(row)
        if bridge_intervals is None:
            ground_segments.append(row)
            continue
        for start_f, end_f, is_bridge in bridge_intervals:
            if end_f - start_f < 1e-6:
                continue
            sub_geom = shapely.ops.substring(row.geom, start_f, end_f, normalized=True)
            if is_bridge and sub_geom.length >= MIN_BRIDGE_LEN:
                bridge_segments.append((sub_geom, row))
                raw_bridge_records.append((sub_geom, row))
                continue
            sub_row = row.copy()
            sub_row['geom'] = sub_geom
            ground_segments.append(sub_row)
    segments_df = pd.DataFrame(ground_segments).reset_index(drop=True)

    ROAD_WIDTHS = {
        'motorway': 18, 'trunk': 12, 'primary': 10, 'secondary': 9, 'tertiary': 8,
        'residential': 6, 'unclassified': 6, 'living_street': 5, 'service': 3.6,
        'pedestrian': 4, 'track': 3, 'footway': 1.8, 'path': 1.6, 'cycleway': 2,
        'steps': 2, 'bridleway': 1.6, 'unknown': 3,
    }

    def get_road_width(row):
        width = ROAD_WIDTHS.get(row['class'], 3)
        if row['class'] not in ('motorway', 'trunk', 'primary', 'secondary', 'tertiary'):
            return width
        if str(row.get('subclass')) == 'link':
            return min(width, 5.0)
        restrictions = row.get('access_restrictions')
        is_oneway = (
            restrictions is not None
            and not isinstance(restrictions, float)
            and any(
                r.get('access_type') == 'denied'
                and (r.get('when') or {}).get('heading') == 'backward'
                and r.get('between') is None
                for r in restrictions
            )
        )
        return round(width * 0.6, 1) if is_oneway else width

    ROAD_SURFACE_MAP = {
        'track': 'dirt', 'path': 'dirt', 'bridleway': 'dirt', 'footway': 'footway',
        'cycleway': 'footway', 'steps': 'paved', 'pedestrian': 'cobble',
    }
    DRIVABLE_CLASSES = {
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'residential', 'unclassified', 'living_street', 'service',
    }
    PEDESTRIAN_CLASSES = {'footway', 'path', 'cycleway', 'steps', 'pedestrian', 'bridleway', 'track'}
    MAX_BRIDGE_GRADE = {'road': 0.06, 'rail': 0.03, 'foot': 0.08}

    road_entries = []
    for _, row in segments_df.iterrows():
        road_cls = row['class']
        if row.subtype == 'rail':
            road_entries.append((0, row.geom, 'rail', 3, get_primary_name(row)))
        elif road_cls in ROAD_WIDTHS:
            width = get_road_width(row)
            road_entries.append((width, row.geom, road_cls, width, get_primary_name(row)))

    road_entries = [
        (sort_w, line, r_cls, width, name)
        for sort_w, geom, r_cls, width, name in road_entries
        for line in extract_linestrings(geom)
    ]
    road_spatial_tree = shapely.STRtree([entry[1] for entry in road_entries])
    water_buffer_union = shapely.unary_union([r.geom.buffer(4) for _, r in water_df.iterrows()] or [Point(1e9, 1e9)])
    visited_roads, road_ramp_cuts = set(), {}

    def trace_bridge_approach(start_pt, dir_vec, bridge_kind):
        approach_chain, total_dist = [], 0.0
        for _ in range(8):
            best_match = None
            for idx in road_spatial_tree.query(Point(start_pt).buffer(1.0)):
                idx = int(idx)
                if idx in visited_roads:
                    continue
                _, line, r_cls, _, _ = road_entries[idx]
                if (r_cls == 'rail') != (bridge_kind == 'rail'):
                    continue
                if bridge_kind != 'rail' and (r_cls in PEDESTRIAN_CLASSES) != (bridge_kind == 'foot'):
                    continue
                if line.length < 0.5 or line.distance(Point(start_pt)) > 1.0:
                    continue
                proj_dist = line.project(Point(start_pt))
                for forward in (True, False):
                    rem_len = line.length - proj_dist if forward else proj_dist
                    if rem_len < 0.5:
                        continue
                    step_pt = line.interpolate(proj_dist + min(5.0, rem_len) if forward else proj_dist - min(5.0, rem_len))
                    anchor_pt = line.interpolate(proj_dist)
                    vx, vy = step_pt.x - anchor_pt.x, step_pt.y - anchor_pt.y
                    norm = math.hypot(vx, vy) or 1.0
                    cos_angle = (vx * dir_vec[0] + vy * dir_vec[1]) / norm
                    min_cos = {'road': 0.5, 'rail': 0.7, 'foot': -0.1}[bridge_kind]
                    if cos_angle > min_cos and (best_match is None or cos_angle > best_match[0]):
                        best_match = (cos_angle, idx, forward, proj_dist)
            if best_match is None:
                break
            _, idx, forward, proj_dist = best_match
            visited_roads.add(idx)
            approach_chain.append((idx, forward, proj_dist))
            line = road_entries[idx][1]
            total_dist += (line.length - proj_dist) if forward else proj_dist
            end_coord = line.coords[-1] if forward else line.coords[0]
            prev_pt = line.interpolate(max(proj_dist, line.length - 3)) if forward else line.interpolate(min(proj_dist, 3.0))
            vx, vy = end_coord[0] - prev_pt.x, end_coord[1] - prev_pt.y
            norm = math.hypot(vx, vy) or 1.0
            dir_vec = (vx / norm, vy / norm)
            start_pt = end_coord[:2]
            if total_dist >= 200:
                break
        return approach_chain

    def extract_oriented_coords(road_idx, forward, proj_dist):
        line = road_entries[road_idx][1]
        sub = shapely.ops.substring(line, proj_dist, line.length) if forward else shapely.ops.substring(line, 0, proj_dist)
        coords = [c[:2] for c in sub.coords]
        return coords if forward else coords[::-1]

    def apply_ramp_cuts(approach_chain, cut_length):
        for road_idx, forward, proj_dist in approach_chain:
            line = road_entries[road_idx][1]
            avail_len = line.length - proj_dist if forward else proj_dist
            c_start, c_end = (proj_dist, min(line.length, proj_dist + cut_length)) if forward else (max(0.0, proj_dist - cut_length), proj_dist)
            if c_end - c_start > 0.3:
                road_ramp_cuts.setdefault(road_idx, []).append((c_start, c_end))
            cut_length -= avail_len
            if cut_length <= 0.3:
                break

    def classify_bridge_kind(row):
        if row.subtype == 'rail':
            return 'rail'
        return 'foot' if row['class'] in PEDESTRIAN_CLASSES else 'road'

    grouped_bridges = {}
    for geom, row in bridge_segments:
        grouped_bridges.setdefault(classify_bridge_kind(row), []).append((geom, row))

    merged_bridge_segments = []
    for _, items in grouped_bridges.items():
        multi_line = shapely.MultiLineString([sub_l for geom, _ in items for sub_l in extract_linestrings(geom)])
        for merged_line in extract_linestrings(shapely.ops.linemerge(multi_line)):
            best_row = max(items, key=lambda it: it[0].intersection(merged_line.buffer(0.5)).length)[1]
            merged_bridge_segments.append((merged_line, best_row))

    bridge_decks = []
    for geom, row in merged_bridge_segments:
        for line in extract_linestrings(geom):
            if not line.intersects(draw_bounds):
                continue
            road_cls = row['class']
            kind = classify_bridge_kind(row)
            deck_width = 4.5 if kind == 'rail' else get_road_width(row)
            bridge_coords = [c[:2] for c in line.coords]

            def unit_vec(pt_a, pt_b):
                dist = math.dist(pt_a, pt_b) or 1.0
                return ((pt_a[0] - pt_b[0]) / dist, (pt_a[1] - pt_b[1]) / dist)

            visited_roads.clear()
            pt_near_start = line.interpolate(min(3.0, line.length / 2))
            pt_near_end = line.interpolate(max(0.0, line.length - 3))
            app_start = trace_bridge_approach(bridge_coords[0], unit_vec(bridge_coords[0], (pt_near_start.x, pt_near_start.y)), kind)
            app_end = trace_bridge_approach(bridge_coords[-1], unit_vec(bridge_coords[-1], (pt_near_end.x, pt_near_end.y)), kind)

            pre_coords = [c for idx, fwd, t in app_start for c in extract_oriented_coords(idx, fwd, t)[1:]]
            post_coords = [c for idx, fwd, t in app_end for c in extract_oriented_coords(idx, fwd, t)[1:]]
            full_profile_line = LineString(pre_coords[::-1] + bridge_coords + post_coords)

            span_start_d = LineString(pre_coords[::-1] + bridge_coords[:1]).length if pre_coords else 0.0
            span_end_d = span_start_d + line.length
            dense_line = full_profile_line.segmentize(4.0)
            xy_pts = np.array(dense_line.coords)[:, :2]
            cum_dist = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy_pts, axis=0).T))])
            total_profile_len = cum_dist[-1]

            ground_z = get_ground_elevation(xy_pts[:, 0], xy_pts[:, 1])
            is_on_span = (cum_dist >= span_start_d - 0.01) & (cum_dist <= span_end_d + 0.01)
            required_z = ground_z + 0.15

            endpoints_buf = line.boundary.buffer(1.5)
            crossed_geoms = [
                road_spatial_tree.geometries[j]
                for j in road_spatial_tree.query(line)
                if road_spatial_tree.geometries[j].intersects(line) and not road_spatial_tree.geometries[j].intersects(endpoints_buf)
            ]
            crossed_clearance_zone = shapely.unary_union(crossed_geoms).buffer(deck_width / 2 + 3) if crossed_geoms else None

            for i, (px, py) in enumerate(xy_pts):
                if not is_on_span[i]:
                    continue
                pt = Point(px, py)
                if crossed_clearance_zone is not None and crossed_clearance_zone.contains(pt):
                    required_z[i] = ground_z[i] + (6.0 if kind == 'rail' else 5.2)
                elif water_buffer_union.contains(pt):
                    required_z[i] = ground_z[i] + (1.0 if kind == 'foot' else 1.3)

            max_grade = MAX_BRIDGE_GRADE[kind]
            height_above_ground = required_z - ground_z
            graded_height = np.max(height_above_ground[None, :] - max_grade * np.abs(cum_dist[:, None] - cum_dist[None, :]), axis=1)
            graded_height = np.minimum(graded_height, 0.15 + 0.2 * np.minimum(cum_dist, total_profile_len - cum_dist))
            profile_z = ground_z + np.maximum(graded_height, 0.15)

            # Upper convex hull smoothing across bridge span
            span_indices = np.where(is_on_span)[0]
            upper_hull = []
            for pt_dz in zip(cum_dist[span_indices], profile_z[span_indices]):
                while (
                    len(upper_hull) >= 2
                    and (upper_hull[-1][0] - upper_hull[-2][0]) * (pt_dz[1] - upper_hull[-2][1])
                    - (upper_hull[-1][1] - upper_hull[-2][1]) * (pt_dz[0] - upper_hull[-2][0]) >= 0
                ):
                    upper_hull.pop()
                upper_hull.append(pt_dz)
            profile_z[span_indices] = np.interp(cum_dist[span_indices], [q[0] for q in upper_hull], [q[1] for q in upper_hull])

            elevated_indices = np.where(profile_z - ground_z > 0.25)[0]
            start_idx = min(max(int(elevated_indices[0]) - 1, 0) if len(elevated_indices) else span_indices[0], span_indices[0])
            end_idx = max(min(int(elevated_indices[-1]) + 1, len(cum_dist) - 1) if len(elevated_indices) else span_indices[-1], span_indices[-1])

            apply_ramp_cuts(app_start, span_start_d - cum_dist[start_idx])
            apply_ramp_cuts(app_end, cum_dist[end_idx] - span_end_d)

            trimmed_line = LineString(xy_pts[start_idx:end_idx + 1])
            trimmed_dist = cum_dist[start_idx:end_idx + 1] - cum_dist[start_idx]
            bridge_decks.append((
                trimmed_line,
                trimmed_dist,
                profile_z[start_idx:end_idx + 1],
                ground_z[start_idx:end_idx + 1],
                deck_width,
                kind,
                road_cls,
                get_primary_name(row),
                int(line.intersects(water_buffer_union)),
            ))

    for custom_bridge in manual_overrides.get('bridges', []):
        b_line = LineString([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in custom_bridge['line']]).segmentize(2.0)
        if b_line.intersects(draw_bounds):
            b_xy = np.array(b_line.coords)[:, :2]
            b_dist = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(b_xy, axis=0).T))])
            b_ground_z = get_ground_elevation(b_xy[:, 0], b_xy[:, 1])
            norm_t = b_dist / max(b_dist[-1], 0.1)
            b_z = b_ground_z + 0.35 + custom_bridge.get('h', 1.2) * np.sin(np.pi * norm_t)
            bridge_decks.append((
                b_line, b_dist, b_z, b_ground_z,
                custom_bridge.get('w', 2.5),
                custom_bridge.get('k', 'foot'),
                custom_bridge.get('c', 'footway'),
                None,
                1,
            ))

    print('mostů', len(bridge_decks), 'z toho lávek', sum(d[5] == 'foot' for d in bridge_decks), 'nájezdů', len(road_ramp_cuts))

    def trim_road_cuts(line, cuts):
        remaining_parts, cursor = [], 0.0
        for cut_start, cut_end in sorted(cuts):
            if cut_start - cursor > 0.3:
                remaining_parts.append(shapely.ops.substring(line, cursor, cut_start))
            cursor = max(cursor, cut_end)
        if line.length - cursor > 0.3:
            remaining_parts.append(shapely.ops.substring(line, cursor, line.length))
        return remaining_parts

    road_entries = [
        (sort_w, sub_line, r_cls, width, name)
        for idx, (sort_w, line, r_cls, width, name) in enumerate(road_entries)
        for sub_line in (trim_road_cuts(line, road_ramp_cuts[idx]) if idx in road_ramp_cuts else [line])
    ]
    road_entries.sort(key=lambda entry: entry[0])

    # ---------------- Sidewalks & Road Surface Painting ----------------
    print("Sidewalks & Road Surface Painting")
    SIDEWALK_ROADS = {'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified'}
    footpath_tree = shapely.STRtree([
        sub_l
        for _, geom, r_cls, _, _ in road_entries
        if r_cls in ('footway', 'pedestrian', 'path', 'cycleway', 'steps')
        for sub_l in extract_linestrings(geom)
    ])
    early_building_tree = shapely.STRtree([poly for geom in buildings_df.geom for poly in extract_polygons(geom)])

    for _, geom, r_cls, width, _ in road_entries:
        if r_cls not in SIDEWALK_ROADS:
            continue
        for line in extract_linestrings(geom.intersection(draw_bounds)):
            for side in (-1, 1):
                try:
                    offset_geom = line.offset_curve(side * (width / 2 + 1.05))
                except Exception:
                    continue
                for offset_line in extract_linestrings(offset_geom):
                    cursor = 0.0
                    while cursor < offset_line.length:
                        segment_piece = shapely.ops.substring(offset_line, cursor, min(cursor + 6, offset_line.length))
                        midpoint = segment_piece.interpolate(0.5, normalized=True)
                        near_footpaths = footpath_tree.query(midpoint.buffer(3.0))
                        near_buildings = early_building_tree.query(midpoint.buffer(0.9))
                        if len(near_footpaths) == 0 and len(near_buildings) == 0 and segment_piece.length > 0.5:
                            paint_line(segment_piece, SURFACE_IDS['footway'], 1.7, block_trees=True)
                        cursor += 6

    for _, geom, r_cls, width, _ in road_entries:
        if r_cls != 'rail' and width >= 3.5:
            paint_line(geom, SURFACE_IDS['curb'], width + 0.7, block_trees=True)

    for _, geom, r_cls, width, _ in road_entries:
        if r_cls == 'rail':
            paint_line(geom, SURFACE_IDS['dirt'], 3.2, block_trees=True)
            paint_line(geom, SURFACE_IDS['rail'], 1.4)
        else:
            paint_line(geom, SURFACE_IDS[ROAD_SURFACE_MAP.get(r_cls, 'road')], width, block_trees=width >= 3)

    for custom_area in manual_overrides.get('plochy', []):
        area_poly = Polygon([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in custom_area['q']])
        paint_polygon(area_poly, SURFACE_IDS[custom_area['k']])

    # Road center dashed lines, manholes, and yield lines
    for _, geom, r_cls, width, _ in road_entries:
        if r_cls in ('trunk', 'primary', 'secondary', 'tertiary') or (r_cls == 'motorway' and width >= 7):
            for line in extract_linestrings(geom.intersection(chunk_box)):
                cursor = 0.0
                while cursor < line.length - 3:
                    dash_pts = [line.interpolate(cursor).coords[0], line.interpolate(cursor + 3).coords[0]]
                    surface_draw.line(to_tex_coords(dash_pts), fill=SURFACE_IDS['marking'], width=1)
                    cursor += 9

    drivable_lines = [
        (sub_l, width, r_cls, name)
        for _, geom, r_cls, width, name in road_entries
        if r_cls in DRIVABLE_CLASSES
        for sub_l in extract_linestrings(geom)
    ]
    ROAD_HIERARCHY = {
        'service': 0, 'living_street': 1, 'residential': 2, 'unclassified': 2,
        'tertiary': 3, 'secondary': 4, 'primary': 5, 'trunk': 6, 'motorway': 7,
    }
    endpoint_ranks = {}
    for line, _, r_cls, _ in drivable_lines:
        for endpoint in (line.coords[0], line.coords[-1]):
            endpoint_ranks.setdefault((round(endpoint[0]), round(endpoint[1])), []).append(ROAD_HIERARCHY[r_cls])

    for line, width, r_cls, _ in drivable_lines:
        if line.length < 10 or not chunk_box.buffer(-5).intersects(line):
            continue
        if width >= 5:
            cursor = 18.0
            while cursor < line.length - 5:
                pt = line.interpolate(cursor)
                pt_prev = line.interpolate(cursor - 1)
                pt_next = line.interpolate(cursor + 1)
                heading = math.atan2(pt_next.y - pt_prev.y, pt_next.x - pt_prev.x)
                mh_x = pt.x + math.cos(heading - math.pi / 2) * width / 4
                mh_y = pt.y + math.sin(heading - math.pi / 2) * width / 4
                radius_px = 0.38 / tex_meters_per_px
                tex_cx, tex_cy = to_tex_coords([(mh_x, mh_y)])[0]
                surface_draw.ellipse(
                    [tex_cx - radius_px, tex_cy - radius_px, tex_cx + radius_px, tex_cy + radius_px],
                    fill=SURFACE_IDS['manhole'],
                )
                cursor += 38
        for is_end in (0, 1):
            endpoint = line.coords[-1] if is_end else line.coords[0]
            ranks = endpoint_ranks.get((round(endpoint[0]), round(endpoint[1])), [])
            if len(ranks) < 3 or max(ranks) <= ROAD_HIERARCHY[r_cls] or ROAD_HIERARCHY[r_cls] == 0:
                continue
            stop_dist = line.length - 3.5 if is_end else 3.5
            pt = line.interpolate(stop_dist)
            pt_prev = line.interpolate(max(0, stop_dist - 1))
            pt_next = line.interpolate(min(line.length, stop_dist + 1))
            tx, ty = pt_next.x - pt_prev.x, pt_next.y - pt_prev.y
            t_len = math.hypot(tx, ty) or 1.0
            tx, ty = tx / t_len, ty / t_len
            if not is_end:
                tx, ty = -tx, -ty
            right_x, right_y = ty, -tx
            yield_pts = [(pt.x, pt.y), (pt.x + right_x * (width / 2 - 0.3), pt.y + right_y * (width / 2 - 0.3))]
            surface_draw.line(
                to_tex_coords(yield_pts),
                fill=SURFACE_IDS['marking'],
                width=max(1, round(0.45 / tex_meters_per_px)),
            )

    non_rail_lines = [
        (sub_l, width, r_cls, name)
        for _, geom, r_cls, width, name in road_entries
        if r_cls != 'rail'
        for sub_l in extract_linestrings(geom)
    ]

    def clip_deck_runs(pts_3d):
        runs, current_run = [], []
        for pt in pts_3d:
            if extended_area_prep.contains(Point(pt[0], pt[1])):
                current_run.append(pt)
            else:
                if len(current_run) > 1:
                    runs.append(current_run)
                current_run = []
        if len(current_run) > 1:
            runs.append(current_run)
        return runs

    def serialize_deck_line(deck):
        deck_line, _, deck_z = deck[0], deck[1], deck[2]
        return [
            [round(x, 1), round(y, 1), round(float(z), 2)]
            for (x, y), z in zip(np.array(deck_line.coords)[:, :2], deck_z)
        ]

    drivable_tree = shapely.STRtree([item[0] for item in drivable_lines])
    non_rail_tree = shapely.STRtree([item[0] for item in non_rail_lines])

    def find_nearest_road_direction(pt, tree, line_items, max_dist):
        nearest_idx = tree.nearest(pt)
        if nearest_idx is None:
            return None
        line = line_items[nearest_idx][0]
        if line.distance(pt) > max_dist:
            return None
        proj = line.project(pt)
        pt_a = line.interpolate(max(0, proj - 1))
        pt_b = line.interpolate(min(line.length, proj + 1))
        return math.atan2(pt_b.y - pt_a.y, pt_b.x - pt_a.x), line_items[nearest_idx]

    # Zebra crosswalk markings
    for _, row in infra_df.iterrows():
        if row['class'] != 'crossing' or not chunk_box.contains(row.geom):
            continue
        nearest_road = find_nearest_road_direction(row.geom, drivable_tree, drivable_lines, 6)
        if not nearest_road:
            continue
        heading, (line, width, *_) = nearest_road
        center_pt = line.interpolate(line.project(row.geom))
        ux, uy = math.cos(heading), math.sin(heading)
        vx, vy = -uy, ux
        offset = -width / 2 + 0.5
        while offset < width / 2 - 0.3:
            stripe_cx, stripe_cy = center_pt.x + vx * offset, center_pt.y + vy * offset
            stripe_corners = [
                (stripe_cx + ux * s * 1.5 + vx * t * 0.25, stripe_cy + uy * s * 1.5 + vy * t * 0.25)
                for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
            surface_draw.polygon(to_tex_coords(stripe_corners), fill=SURFACE_IDS['marking'])
            offset += 1.0

    # ---------------- Parked Cars ----------------
    print("Parked Cars")
    parked_cars = []
    service_aisles = [sub_l for _, geom, r_cls, _, _ in road_entries if r_cls == 'service' for sub_l in extract_linestrings(geom)]
    aisle_tree = shapely.STRtree(service_aisles) if service_aisles else None
    occupied_stalls = {}

    def claim_parking_stall(x, y):
        cell_x, cell_y = int(x // 3), int(y // 3)
        for nx in range(cell_x - 1, cell_x + 2):
            for ny in range(cell_y - 1, cell_y + 2):
                for existing_pt in occupied_stalls.get((nx, ny), []):
                    if (existing_pt[0] - x) ** 2 + (existing_pt[1] - y) ** 2 < 4.0:
                        return False
        occupied_stalls.setdefault((cell_x, cell_y), []).append((x, y))
        return True

    for parking_id, lot_poly in parking_polygons:
        if lot_poly.area < 120:
            continue
        lot_rng = random.Random(f'{parking_id}{world_coord_key(lot_poly.centroid.x, lot_poly.centroid.y, center_x, center_y)}')
        aisles_in_lot = []
        if aisle_tree is not None:
            buffered_lot = lot_poly.buffer(0.5)
            for idx in sorted(int(v) for v in aisle_tree.query(lot_poly)):
                aisles_in_lot += [sub_l for sub_l in extract_linestrings(service_aisles[idx].intersection(buffered_lot)) if sub_l.length >= 6]

        if aisles_in_lot:
            inner_lot_prep = shapely.prepared.prep(lot_poly.buffer(-0.3))
            local_aisle_tree = shapely.STRtree(aisles_in_lot)
            for aisle_line in aisles_in_lot:
                cursor = 1.25
                while cursor < aisle_line.length - 1.25:
                    p_prev = aisle_line.interpolate(cursor - 1.0)
                    p_next = aisle_line.interpolate(cursor + 1.0)
                    p_curr = aisle_line.interpolate(cursor)
                    seg_len = math.hypot(p_next.x - p_prev.x, p_next.y - p_prev.y) or 1.0
                    tx, ty = (p_next.x - p_prev.x) / seg_len, (p_next.y - p_prev.y) / seg_len
                    for side in (1, -1):
                        nx, ny = -ty * side, tx * side
                        pt_front, pt_back, pt_center = [Point(p_curr.x + nx * d, p_curr.y + ny * d) for d in (3.0, 8.0, 5.5)]
                        if not (inner_lot_prep.contains(pt_front) and inner_lot_prep.contains(pt_back) and inner_lot_prep.contains(pt_center)):
                            continue
                        if len(local_aisle_tree.query(pt_center, predicate='dwithin', distance=4.6)) or not claim_parking_stall(pt_center.x, pt_center.y):
                            continue
                        surface_draw.line(
                            to_tex_coords([(pt_front.x - tx * 1.25, pt_front.y - ty * 1.25), (pt_back.x - tx * 1.25, pt_back.y - ty * 1.25)]),
                            fill=SURFACE_IDS['marking'],
                            width=1,
                        )
                        surface_draw.line(
                            to_tex_coords([(pt_front.x + tx * 1.25, pt_front.y + ty * 1.25), (pt_back.x + tx * 1.25, pt_back.y + ty * 1.25)]),
                            fill=SURFACE_IDS['marking'],
                            width=1,
                        )
                        if lot_rng.random() < 0.58:
                            parked_cars.append([pt_center.x, pt_center.y, math.atan2(ny, nx), lot_rng.randrange(8)])
                    cursor += 2.5
            continue

        obb = lot_poly.minimum_rotated_rectangle
        corners = [np.array(c) for c in list(obb.exterior.coords)[:4]]
        edge1, edge2 = corners[1] - corners[0], corners[3] - corners[0]
        if np.linalg.norm(edge1) < np.linalg.norm(edge2):
            edge1, edge2 = edge2, edge1
            corners = [corners[0], corners[3], corners[2], corners[1]]
        lot_len, lot_width = np.linalg.norm(edge1), np.linalg.norm(edge2)
        unit_u, unit_v = edge1 / lot_len, edge2 / lot_width
        stall_rows = [(0, 5)] if lot_width < 11 else [(0, 5), (lot_width - 5, lot_width)]
        if lot_width >= 22:
            stall_rows += [(lot_width / 2 - 5, lot_width / 2), (lot_width / 2, lot_width / 2 + 5)]
        inner_lot_prep = shapely.prepared.prep(lot_poly.buffer(-0.3))
        for r0, r1 in stall_rows:
            cursor = 0.5
            while cursor + 2.5 < lot_len:
                pt_a = corners[0] + unit_u * cursor + unit_v * r0
                pt_b = corners[0] + unit_u * cursor + unit_v * r1
                if inner_lot_prep.contains(Point(*pt_a)) and inner_lot_prep.contains(Point(*pt_b)):
                    surface_draw.line(to_tex_coords([tuple(pt_a), tuple(pt_b)]), fill=SURFACE_IDS['marking'], width=1)
                pt_c = corners[0] + unit_u * (cursor + 1.25) + unit_v * (r0 + r1) / 2
                if inner_lot_prep.contains(Point(*pt_c)) and lot_rng.random() < 0.58:
                    parked_cars.append([float(pt_c[0]), float(pt_c[1]), math.atan2(unit_v[1], unit_v[0]), lot_rng.randrange(8)])
                cursor += 2.5

    rail_tree = shapely.STRtree([sub_l for _, geom, r_cls, _, _ in road_entries if r_cls == 'rail' for sub_l in extract_linestrings(geom)] or [Point(1e9, 1e9)])
    for line, width, r_cls, _ in drivable_lines:
        if r_cls not in ('residential', 'unclassified', 'living_street', 'tertiary'):
            continue
        cursor = 4.0
        street_rng = random.Random('car' + world_coord_key(*line.coords[0], center_x, center_y))
        while cursor < line.length - 4:
            if street_rng.random() < 0.07:
                pt_prev = line.interpolate(cursor - 1)
                pt_next = line.interpolate(cursor + 1)
                heading = math.atan2(pt_next.y - pt_prev.y, pt_next.x - pt_prev.x)
                curb_offset = width / 2 - 1.1
                pt = line.interpolate(cursor)
                car_x = pt.x + math.cos(heading - math.pi / 2) * curb_offset
                car_y = pt.y + math.sin(heading - math.pi / 2) * curb_offset
                if chunk_box.contains(Point(car_x, car_y)):
                    car_variant = street_rng.randrange(8)
                    if not len(rail_tree.query(Point(car_x, car_y), predicate='dwithin', distance=8.0)):
                        parked_cars.append([car_x, car_y, heading, car_variant])
            cursor += 7

    # ---------------- Buildings & Landmarks ----------------
    print("Buildings & Landmarks")
    buildings_df = buildings_df[
        [(srcs is not None and any(s['dataset'] == 'OpenStreetMap' for s in srcs)) or b_id in manual_overrides['b']
         for srcs, b_id in zip(buildings_df.sources, buildings_df.id)]
    ].reset_index(drop=True)

    for idx, row in buildings_df.iterrows():
        if 'h' in manual_overrides['b'].get(row.id, {}):
            buildings_df.at[idx, 'height'] = manual_overrides['b'][row.id]['h']

    if len(buildings_df) and manual_overrides['roofs']:
        template_row = buildings_df.iloc[0]
        extra_rows = []
        for idx, roof_cfg in enumerate(manual_overrides['roofs']):
            roof_poly = LineString([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in roof_cfg['line']]).buffer(
                roof_cfg['w'] / 2, cap_style='flat', join_style='mitre'
            )
            extra_rows.append({
                **template_row.to_dict(),
                'id': f'up-roof-{idx}',
                'class': 'roof',
                'subtype': None,
                'names': None,
                'height': roof_cfg['h'],
                'num_floors': float('nan'),
                'roof_shape': None,
                'geom': roof_poly,
                'geometry': None,
            })
        buildings_df = pd.concat([buildings_df, pd.DataFrame(extra_rows)], ignore_index=True)

    if len(buildings_df) and manual_overrides.get('custom_bld'):
        template_row = buildings_df.iloc[0]
        extra_rows = []
        for bld_cfg in manual_overrides['custom_bld']:
            bld_poly = Polygon([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in bld_cfg['poly']])
            satellite_data['b'][bld_cfg['id']] = {k: bld_cfg[k] for k in ('h', 'r', 'wc', 'rc') if k in bld_cfg}
            extra_rows.append({
                **template_row.to_dict(),
                'id': bld_cfg['id'],
                'class': bld_cfg.get('cls', 'service'),
                'subtype': None,
                'names': {'primary': bld_cfg.get('name')} if bld_cfg.get('name') else None,
                'height': bld_cfg['h'],
                'num_floors': float('nan'),
                'roof_shape': bld_cfg.get('r', 'flat'),
                'geom': bld_poly,
                'geometry': None,
            })
        buildings_df = pd.concat([buildings_df, pd.DataFrame(extra_rows)], ignore_index=True)

    STANDARD_WALL_PALETTE = [
        (232, 220, 192), (241, 234, 216), (217, 201, 163), (230, 211, 179),
        (214, 208, 196), (239, 230, 207), (216, 185, 143), (226, 196, 160), (205, 214, 206),
    ]
    PANEL_WALL_PALETTE = [
        (232, 226, 208), (213, 217, 222), (236, 236, 230), (222, 224, 222),
        (226, 199, 154), (169, 195, 201), (238, 232, 218), (204, 210, 214),
    ]
    PITCHED_ROOF_PALETTE = [
        (168, 68, 47), (150, 60, 44), (142, 58, 42), (107, 59, 47),
        (74, 74, 82), (63, 74, 90), (156, 90, 60), (128, 52, 40),
    ]
    FLAT_ROOF_PALETTE = [(141, 141, 141), (122, 122, 122), (156, 154, 146), (110, 112, 116)]
    TENEMENT_WALL_PALETTE = [
        (232, 206, 150), (226, 186, 160), (206, 214, 180), (236, 226, 200), (214, 196, 170),
        (204, 202, 196), (238, 214, 170), (220, 176, 146), (196, 208, 214), (230, 220, 210),
    ]
    TENEMENT_ROOF_PALETTE = [(70, 72, 80), (96, 64, 52), (150, 62, 44), (60, 64, 72), (128, 60, 46), (84, 88, 96)]

    PITCHED_CLASSES = {
        'church', 'house', 'detached', 'semidetached_house', 'residential', 'terrace', 'farm', 'chapel',
        'bungalow', 'villa', 'cabin', 'religious', 'civic', 'hut', 'barn', 'farm_auxiliary',
        'kindergarten', 'allotment_house',
    }
    FLAT_CLASSES = {
        'garage', 'garages', 'apartments', 'industrial', 'warehouse', 'commercial', 'retail', 'school',
        'university', 'hospital', 'office', 'carport', 'parking', 'service', 'transportation',
        'sports_hall', 'sports_centre', 'greenhouse', 'roof',
    }
    ROOF_SHAPE_MAP = {
        'gabled': 'gabled', 'hipped': 'hipped', 'half_hipped': 'hipped', 'pyramidal': 'pyramidal',
        'saltbox': 'gabled', 'gambrel': 'gabled', 'skillion': 'gabled', 'mansard': 'hipped',
        'flat': 'flat', 'round': 'flat', 'dome': 'pyramidal',
    }
    CZECH_BUILDING_LABELS = {
        'house': 'rodinný dům', 'detached': 'rodinný dům', 'residential': 'obytný dům', 'apartments': 'bytový dům',
        'garage': 'garáž', 'garages': 'garáže', 'industrial': 'průmyslová budova', 'warehouse': 'sklad',
        'civic': 'občanská stavba', 'school': 'škola', 'university': 'univerzita', 'retail': 'obchod',
        'commercial': 'komerční budova', 'church': 'kostel', 'chapel': 'kaple', 'kindergarten': 'mateřská škola',
        'shed': 'kůlna', 'barn': 'stodola', 'service': 'technická budova', 'office': 'kanceláře',
        'semidetached_house': 'dvojdům', 'terrace': 'řadový dům', 'bungalow': 'bungalov', 'hotel': 'hotel',
        'dormitory': 'kolej', 'hospital': 'zdravotnictví', 'greenhouse': 'skleník', 'roof': 'přístřešek',
        'cabin': 'chata', 'farm': 'usedlost', 'farm_auxiliary': 'hospodářská budova', 'transportation': 'stavba pro dopravu',
    }

    def parse_color_rgb(color_str, fallback_rgb):
        if not isinstance(color_str, str):
            return fallback_rgb
        color_str = color_str.strip()
        if color_str.startswith('#') and len(color_str) == 7:
            try:
                return tuple(int(color_str[i:i + 2], 16) for i in (1, 3, 5))
            except ValueError:
                return fallback_rgb
        named_colors = {
            'white': (240, 238, 232), 'red': (168, 68, 47), 'brown': (110, 70, 50),
            'grey': (130, 130, 130), 'gray': (130, 130, 130), 'black': (60, 60, 64),
            'yellow': (232, 208, 140), 'beige': (226, 210, 176), 'darkred': (128, 46, 36),
            'green': (80, 120, 80), 'blue': (80, 100, 140), 'orange': (204, 120, 60),
        }
        return named_colors.get(color_str.lower(), fallback_rgb)

    def find_largest_grid_rect(mask_grid):
        best = (0, None)
        heights = np.zeros(mask_grid.shape[1], int)
        for row_idx in range(mask_grid.shape[0]):
            heights = np.where(mask_grid[row_idx], heights + 1, 0)
            stack = []
            for col_idx in range(mask_grid.shape[1] + 1):
                curr_h = heights[col_idx] if col_idx < mask_grid.shape[1] else 0
                start_col = col_idx
                while stack and stack[-1][1] >= curr_h:
                    s_col, s_h = stack.pop()
                    if s_h * (col_idx - s_col) > best[0]:
                        best = (s_h * (col_idx - s_col), (row_idx - s_h + 1, s_col, row_idx, col_idx - 1))
                    start_col = s_col
                stack.append((start_col, curr_h))
        return best[1]

    def decompose_roof_sections(poly, ridge_angle):
        cell_size = 0.6
        origin_pt = poly.centroid
        aligned_poly = shapely.affinity.rotate(poly, -ridge_angle, origin=origin_pt, use_radians=True)
        minx, miny, maxx, maxy = aligned_poly.bounds
        xs = np.arange(minx + cell_size / 2, maxx, cell_size)
        ys = np.arange(miny + cell_size / 2, maxy, cell_size)
        if len(xs) < 3 or len(ys) < 3:
            return []
        grid_x, grid_y = np.meshgrid(xs, ys)
        inside_mask = shapely.contains_xy(aligned_poly, grid_x, grid_y)
        total_cells = inside_mask.sum()
        sections = []
        for _ in range(4):
            rect = find_largest_grid_rect(inside_mask)
            if rect is None:
                break
            r0, c0, r1, c1 = rect
            rect_cells = (r1 - r0 + 1) * (c1 - c0 + 1)
            if rect_cells < max(0.14 * total_cells, 20 / cell_size ** 2):
                break
            sec_w, sec_h = (c1 - c0 + 1) * cell_size, (r1 - r0 + 1) * cell_size
            if min(sec_w, sec_h) < 3.0:
                inside_mask[r0:r1 + 1, c0:c1 + 1] = False
                continue
            local_cx, local_cy = (xs[c0] + xs[c1]) / 2, (ys[r0] + ys[r1]) / 2
            world_pt = shapely.affinity.rotate(Point(local_cx, local_cy), ridge_angle, origin=origin_pt, use_radians=True)
            sec_angle = ridge_angle if sec_w >= sec_h else ridge_angle + math.pi / 2
            sections.append([
                round(world_pt.x, 2),
                round(world_pt.y, 2),
                round(sec_angle, 4),
                round(max(sec_w, sec_h), 2),
                round(min(sec_w, sec_h), 2),
            ])
            inside_mask[r0:r1 + 1, c0:c1 + 1] = False
        return sections

    address_df = load_spatial_parquet(data_dir, 'address')
    address_df = address_df[address_df.geom.map(lambda g: extended_area_prep.contains(g))]
    try:
        address_points = list(address_df.geom)
    except AttributeError:
        address_points = list()
    address_tree = shapely.STRtree(address_points)
    try:
        address_labels = [
            f"{street} {num}" if isinstance(street, str) else str(num)
            for street, num in zip(address_df.street, address_df.number)
        ]
    except AttributeError:
        address_labels = []

    scene_buildings, building_metadata, residential_chimneys = [], [], []
    building_mask_img = Image.new('L', (ext_tex_width, ext_tex_width), 0)
    building_mask_draw = ImageDraw.Draw(building_mask_img)

    landmarks = [
        d for d in json.load(open('assets/dominanty.json', encoding='utf-8')) if d.get('spec')
    ] if os.path.exists('assets/dominanty.json') else []
    for lm_cfg in landmarks:
        lm_cfg['xy'] = [float(v) for v in geo.convert_lonlat_to_local_meters(lm_cfg['lon'], lm_cfg['lat'])]
        if 'face' in lm_cfg['spec']:
            lm_cfg['spec']['face_xy'] = [float(v) for v in geo.convert_lonlat_to_local_meters(lm_cfg['spec']['face'][1], lm_cfg['spec']['face'][0])]

    landmarks_by_bld_id = {}

    def match_landmarks_to_buildings():
        row_lookup = {i: row for i, row in zip(range(len(buildings_df)), buildings_df.itertuples())}
        geoms = list(buildings_df.geom)
        bld_tree = shapely.STRtree(geoms)
        for lm_cfg in landmarks:
            spec = lm_cfg['spec']
            pt = Point(*lm_cfg['xy'])
            if spec.get('only'):
                continue
            if 'names' in spec:
                for idx, row in row_lookup.items():
                    if isinstance(row.names, dict) and (row.names.get('primary') or '').startswith(spec['names']):
                        landmarks_by_bld_id[row.id] = lm_cfg
                continue
            if 'glass' in spec:
                for idx in bld_tree.query(pt.buffer(spec['glass'])):
                    landmarks_by_bld_id.setdefault(
                        row_lookup[idx].id,
                        {**lm_cfg, 'spec': {'rc': [176, 204, 214], 'r': 'flat', 'win': 14}},
                    )
                continue
            nearby = [idx for idx in bld_tree.query(pt.buffer(25)) if geoms[idx].area > 60]
            containing = [idx for idx in nearby if geoms[idx].buffer(1.0).contains(pt)]
            candidates = containing or sorted(nearby, key=lambda idx: -geoms[idx].area)
            if candidates:
                landmarks_by_bld_id[row_lookup[candidates[0]].id] = lm_cfg

    def compute_landmark_feature_pos(poly, anchor_mode, spec, lm_cfg, feat_width, side_offset=0.0, t_along=0.5):
        centroid = poly.centroid
        if anchor_mode == 'pt':
            return lm_cfg['xy'][0], lm_cfg['xy'][1], 0.0
        obb = poly.minimum_rotated_rectangle
        corners = list(obb.exterior.coords)[:3]
        edge1 = np.subtract(corners[1], corners[0])
        edge2 = np.subtract(corners[2], corners[1])
        major_axis = edge1 if np.linalg.norm(edge1) >= np.linalg.norm(edge2) else edge2
        major_len = float(np.linalg.norm(major_axis))
        major_axis = major_axis / major_len
        axis_angle = math.atan2(major_axis[1], major_axis[0])

        if anchor_mode == 'c':
            return centroid.x + major_axis[0] * side_offset, centroid.y + major_axis[1] * side_offset, axis_angle
        if anchor_mode == 'along':
            dist = (t_along - 0.5) * major_len
            return centroid.x + major_axis[0] * dist, centroid.y + major_axis[1] * dist, axis_angle
        if anchor_mode in ('front', 'back'):
            face_x, face_y = spec.get('face_xy', [centroid.x, centroid.y - 50])
            if anchor_mode == 'back':
                face_x, face_y = 2 * centroid.x - face_x, 2 * centroid.y - face_y
            rect_corners = list(obb.exterior.coords)[:4]
            best_edge = None
            for k in range(4):
                p0, p1 = np.array(rect_corners[k]), np.array(rect_corners[(k + 1) % 4])
                mid = (p0 + p1) / 2
                normal = mid - np.array([centroid.x, centroid.y])
                normal /= np.linalg.norm(normal) or 1.0
                to_face = np.array([face_x - centroid.x, face_y - centroid.y])
                to_face /= np.linalg.norm(to_face) or 1.0
                alignment = float(normal @ to_face)
                if best_edge is None or alignment > best_edge[0]:
                    best_edge = (alignment, p0, p1, mid, normal)
            _, p0, p1, mid, normal = best_edge
            tangent = (p1 - p0) / (np.linalg.norm(p1 - p0) or 1.0)
            pos = mid - normal * (feat_width / 2 + 0.6) + tangent * side_offset
            return float(pos[0]), float(pos[1]), math.atan2(tangent[1], tangent[0])

        compass_dirs = {
            'n': (0, 1), 's': (0, -1), 'e': (1, 0), 'w': (-1, 0),
            'ne': (1, 1), 'nw': (-1, 1), 'se': (1, -1), 'sw': (-1, -1),
        }
        vx, vy = compass_dirs[anchor_mode]
        v_len = math.hypot(vx, vy)
        vx, vy = vx / v_len, vy / v_len
        ext_pts = np.array(poly.exterior.coords)
        furthest_idx = int(np.argmax(ext_pts[:, 0] * vx + ext_pts[:, 1] * vy))
        fx, fy = ext_pts[furthest_idx]
        inset_pt = Point(fx - vx * (feat_width / 2 + 0.4), fy - vy * (feat_width / 2 + 0.4))
        if poly.contains(inset_pt):
            fx, fy = inset_pt.x, inset_pt.y
        else:
            fx, fy = fx + (centroid.x - fx) * 0.15, fy + (centroid.y - fy) * 0.15
        return fx + major_axis[0] * side_offset, fy + major_axis[1] * side_offset, axis_angle

    def split_polygon_holes(poly):
        if not poly.is_valid:
            poly = poly.buffer(0)
        solid_parts, queue = [], extract_polygons(poly)
        while queue:
            curr = queue.pop()
            large_holes = [h for h in curr.interiors if Polygon(h).area > 4]
            if not large_holes or len(solid_parts) > 40:
                solid_parts.append(Polygon(curr.exterior))
                continue
            hole_center = Polygon(large_holes[0]).centroid
            corners = list(curr.minimum_rotated_rectangle.exterior.coords)[:3]
            edge1 = np.subtract(corners[1], corners[0])
            edge2 = np.subtract(corners[2], corners[1])
            cut_dir = edge2 if np.linalg.norm(edge1) >= np.linalg.norm(edge2) else edge1
            cut_dir = cut_dir / (np.linalg.norm(cut_dir) or 1.0) * 1e4
            cut_line = LineString([(hole_center.x - cut_dir[0], hole_center.y - cut_dir[1]),
                                   (hole_center.x + cut_dir[0], hole_center.y + cut_dir[1])])
            split_pieces = [g for g in extract_polygons(shapely.ops.split(curr, cut_line)) if g.area > 1]
            if len(split_pieces) < 2:
                solid_parts.append(Polygon(curr.exterior))
                continue
            queue += split_pieces
        return solid_parts

    def is_compact_obb(poly):
        corners = list(poly.minimum_rotated_rectangle.exterior.coords)[:3]
        side_a, side_b = math.dist(corners[0], corners[1]), math.dist(corners[1], corners[2])
        return max(side_a, side_b) < 70

    match_landmarks_to_buildings()
    print('dominant napárováno na budovy', len(landmarks_by_bld_id))

    all_bld_polys, all_bld_heights, all_bld_lidar = [], [], []
    for b_id, geom, h_val, floors_val in zip(buildings_df.id, buildings_df.geom, buildings_df.height, buildings_df.num_floors):
        for poly in extract_polygons(geom):
            all_bld_lidar.append(lidar_heights[b_id][0] if b_id in lidar_heights else None)
            all_bld_polys.append(poly)
            est_h = h_val if h_val == h_val and h_val else (floors_val * 2.9 if floors_val == floors_val and floors_val else None)
            all_bld_heights.append(est_h)

    bld_poly_tree = shapely.STRtree(all_bld_polys)
    bld_centroid_tree = shapely.STRtree([p.centroid for p in all_bld_polys])
    bld_areas = np.array([p.area for p in all_bld_polys])

    def compute_building_density(pt, radius=70.0):
        circle = pt.buffer(radius)
        nearby_areas = bld_areas[[i for i in bld_centroid_tree.query(circle) if circle.contains(bld_centroid_tree.geometries[i])]]
        return float(nearby_areas.sum()) / circle.area

    def get_connected_group_indices(buffered_poly, limit=60):
        visited = set()
        queue = [i for i in bld_poly_tree.query(buffered_poly) if all_bld_polys[i].intersects(buffered_poly)]
        while queue and len(visited) < limit:
            curr_idx = queue.pop()
            if curr_idx in visited:
                continue
            visited.add(curr_idx)
            adj_buf = all_bld_polys[curr_idx].buffer(0.8)
            queue += [j for j in bld_poly_tree.query(adj_buf) if j not in visited and all_bld_polys[j].intersects(adj_buf)]
        return visited

    def get_connected_lidar_median(buffered_poly, fallback_med):
        group_indices = get_connected_group_indices(buffered_poly)
        vals = [all_bld_lidar[i] for i in group_indices if all_bld_lidar[i]]
        return float(np.median(vals)) if vals else fallback_med

    def get_connected_group_key(buffered_poly):
        group_indices = get_connected_group_indices(buffered_poly)
        min_corner = min((all_bld_polys[i].bounds[0], all_bld_polys[i].bounds[1]) for i in group_indices)
        return world_coord_key(*min_corner, center_x, center_y)

    def get_rect_regularity_and_width(poly):
        corners = list(poly.minimum_rotated_rectangle.exterior.coords)[:3]
        side_a, side_b = math.dist(corners[0], corners[1]), math.dist(corners[1], corners[2])
        return poly.area / max(side_a * side_b, 1e-6), min(side_a, side_b)

    def is_panel_block(buffered_poly):
        neighbors = [i for i in bld_poly_tree.query(buffered_poly) if all_bld_polys[i].intersects(buffered_poly)]
        if len(neighbors) < 2:
            return False
        metrics = [get_rect_regularity_and_width(all_bld_polys[i]) for i in neighbors]
        if min(reg for reg, _ in metrics) < 0.95:
            return False
        widths = [w for _, w in metrics]
        if max(widths) - min(widths) > 1.5 or min(widths) < 8:
            return False
        heights = [all_bld_heights[i] for i in neighbors if all_bld_heights[i]]
        if heights and (max(heights) - min(heights) > 3.5 or min(heights) < 11):
            return False
        return True

    def is_garage_row(poly):
        buf = poly.buffer(0.8)
        visited = set()
        queue = [i for i in bld_poly_tree.query(buf) if bld_areas[i] < 45 and all_bld_polys[i].intersects(buf)]
        while queue and len(visited) < 3:
            curr_idx = queue.pop()
            if curr_idx in visited:
                continue
            visited.add(curr_idx)
            adj_buf = all_bld_polys[curr_idx].buffer(0.8)
            queue += [j for j in bld_poly_tree.query(adj_buf) if j not in visited and bld_areas[j] < 45 and all_bld_polys[j].intersects(adj_buf)]
        return len(visited) >= 3

    def split_by_landmark_zones(pieces, zones):
        out_pieces, zone_assignments = [], []
        for piece in pieces:
            remainder = piece
            for zone in zones:
                if '_P' not in zone:
                    zone['_P'] = Polygon([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in zone['poly']]).buffer(0)
                overlap = remainder.intersection(zone['_P'])
                for sub_g in extract_polygons(overlap):
                    for solid in split_polygon_holes(sub_g):
                        if solid.area > 4:
                            out_pieces.append(solid)
                            zone_assignments.append(zone)
                remainder = remainder.difference(zone['_P'])
            for sub_g in extract_polygons(remainder):
                for solid in split_polygon_holes(sub_g):
                    if solid.area > 4:
                        out_pieces.append(solid)
                        zone_assignments.append(None)
        return out_pieces, zone_assignments

    rooftop_cars = []

    def generate_roof_decorations(poly, deco_list, roof_top_z, bld_key):
        decorations = []
        obb = poly.minimum_rotated_rectangle
        corners = list(obb.exterior.coords)[:3]
        edge1 = np.subtract(corners[1], corners[0])
        edge2 = np.subtract(corners[2], corners[1])
        if np.linalg.norm(edge1) < np.linalg.norm(edge2):
            edge1, edge2 = edge2, edge1
        obb_len, obb_width = float(np.linalg.norm(edge1)), float(np.linalg.norm(edge2))
        unit_u, unit_v = edge1 / obb_len, edge2 / obb_width
        obb_origin = np.array(obb.centroid.coords[0]) - unit_u * obb_len / 2 - unit_v * obb_width / 2

        def rel_pt(s, t):
            return tuple(obb_origin + unit_u * s * obb_len + unit_v * t * obb_width)

        for deco in deco_list:
            kind = deco['k']
            if 'axis' in deco:
                count = deco['axis']
                deco = {**deco, 'w': obb_width / count * deco.get('fill', 1.0)}
                cut_lines = [LineString([rel_pt(-0.1, (i + 0.5) / count), rel_pt(1.1, (i + 0.5) / count)]) for i in range(count)]
            elif 'rel' in deco:
                cut_lines = [LineString([rel_pt(a, b), rel_pt(c, d)]) for a, b, c, d in deco['rel']]
            elif 'lines' in deco:
                cut_lines = [LineString([geo.convert_lonlat_to_local_meters(l[1], l[0]), geo.convert_lonlat_to_local_meters(l[3], l[2])]) for l in deco['lines']]
            else:
                angle = math.radians(deco['a']) if 'a' in deco else math.atan2(unit_u[1], unit_u[0])
                centroid = poly.centroid
                sweep_r = math.sqrt(poly.area) * 3 + 50
                ux, uy = math.cos(angle), math.sin(angle)
                nx, ny = -uy, ux
                cut_lines = [
                    LineString([(centroid.x + nx * t - ux * sweep_r, centroid.y + ny * t - uy * sweep_r),
                                (centroid.x + nx * t + ux * sweep_r, centroid.y + ny * t + uy * sweep_r)])
                    for t in np.arange(-sweep_r + deco['gap'] / 2, sweep_r, deco['gap'])
                ]

            inset_poly = poly.buffer(-max(0.6, deco.get('w', 0) / 2 + 0.3) if kind == 'solar' else -0.6)
            for seg in cut_lines:
                clipped = seg.intersection(inset_poly)
                clipped_lines = [clipped] if clipped.geom_type == 'LineString' else [q for q in getattr(clipped, 'geoms', []) if q.geom_type == 'LineString']
                for sub_l in clipped_lines:
                    if sub_l.length < 1.5:
                        continue
                    (x0, y0), (x1, y1) = sub_l.coords[0], sub_l.coords[-1]
                    if kind == 'park':
                        seg_len = sub_l.length
                        ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
                        park_rng = random.Random(bld_key + f'{x0:.0f}{y0:.0f}')
                        for side in (-1, 1):
                            for t in np.arange(1.6, seg_len - 1.6, 2.6):
                                cx, cy = x0 + ux * t - uy * side * 4.2, y0 + uy * t + ux * side * 4.2
                                if inset_poly.contains(Point(cx, cy)) and park_rng.random() < deco.get('p', 0.72):
                                    car_yaw = math.atan2(ux, -uy) + (math.pi if park_rng.random() < 0.5 else 0)
                                    rooftop_cars.append([cx, cy, roof_top_z, car_yaw, park_rng.randrange(8)])
                        for side in (-1, 1):
                            decorations.append([
                                1, round(x0 - uy * side * 6.6, 2), round(y0 + ux * side * 6.6, 2),
                                round(x1 - uy * side * 6.6, 2), round(y1 + ux * side * 6.6, 2),
                                0.25, 0.04, 226, 226, 220,
                            ])
                            for t in np.arange(0.3, seg_len - 0.3, 2.6):
                                ax, ay = x0 + ux * t - uy * side * 1.8, y0 + uy * t + ux * side * 1.8
                                bx, by = x0 + ux * t - uy * side * 6.6, y0 + uy * t + ux * side * 6.6
                                if inset_poly.contains(Point(ax, ay)) and inset_poly.contains(Point(bx, by)):
                                    decorations.append([1, round(ax, 2), round(ay, 2), round(bx, 2), round(by, 2), 0.2, 0.04, 222, 222, 216])
                        continue
                    kind_code = {'saw': 0, 'strip': 1, 'solar': 2, 'barrel': 3, 'ridge': 4}[kind]
                    deco_w = deco.get('w', deco.get('gap', 4))
                    colors = deco.get('cs', [deco.get('c', [200, 200, 200])])
                    chosen_col = colors[len(decorations) % len(colors)]
                    decorations.append([
                        kind_code, round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2),
                        round(deco_w, 2), deco.get('h', 0.6), *chosen_col, *deco.get('c2', []),
                    ])
        return decorations

    town_squares = [row.geom for _, row in land_use_df.iterrows() if (get_primary_name(row) or '').startswith('náměstí doktora Edvarda Beneše')]
    landmark_ground_z = {}

    for _, row in buildings_df.iterrows():
        if row.is_underground is True:
            continue
        pieces = [solid for geom in extract_polygons(row.geom) for solid in split_polygon_holes(geom)]
        lm_cfg = landmarks_by_bld_id.get(row.id)
        zone_list = [None] * len(pieces)
        shared_lm_ground = None

        if lm_cfg and lm_cfg['spec'].get('zones'):
            all_coords = np.array([c for q in pieces for c in q.exterior.coords])
            shared_lm_ground = float(get_ground_elevation(all_coords[:, 0], all_coords[:, 1]).mean())
            landmark_ground_z[lm_cfg['name']] = shared_lm_ground
            pieces, zone_list = split_by_landmark_zones(pieces, lm_cfg['spec']['zones'])

        tower_piece_idx = None
        if lm_cfg and pieces:
            lm_pt = Point(*lm_cfg['xy'])
            containing = [k for k, q in enumerate(pieces) if q.buffer(2).contains(lm_pt)]
            tower_piece_idx = containing[0] if containing else max(range(len(pieces)), key=lambda k: pieces[k].area)

        for piece_idx, poly in enumerate(pieces):
            if not extended_area.intersects(poly):
                continue
            centroid = poly.centroid
            is_in_core = chunk_box.contains(centroid)
            if not is_in_core and not map_clip_area.contains(centroid):
                continue
            if not map_clip_area.buffer(-1.0).contains(poly):
                poly = poly.intersection(map_clip_area.buffer(-1.0))
            if poly.geom_type != 'Polygon':
                continue
            

            is_urban = geo.find_chunk_containing_point(centroid.x + center_x, centroid.y + center_y) != (0, 0)
            poly = poly.simplify(0.2)
            if poly.area < 6 or not poly.is_valid:
                continue
            poly = shapely.geometry.polygon.orient(poly, 1.0)

            bld_idx = len(scene_buildings)
            bld_cls = row['class'] if isinstance(row['class'], str) else ''
            bld_name = get_primary_name(row)
            osm_id = row.id
            poly_area = poly.area

            ruian_info = ruian_buildings.get(osm_id)
            if ruian_info and ruian_info[0] and bld_cls in ('', 'yes', 'residential'):
                is_flat_lidar = osm_id in lidar_heights and lidar_heights[osm_id][1] - lidar_heights[osm_id][0] < 0.4
                bld_cls = 'residential' if ruian_info[0] == 'apartments' and not is_flat_lidar else ruian_info[0]
            if not bld_cls and poly_area < 45 and is_garage_row(poly):
                bld_cls = 'garages'

            floors = row.num_floors if row.num_floors == row.num_floors and row.num_floors else None
            if not floors and ruian_info and ruian_info[1] and osm_id not in lidar_heights:
                floors = ruian_info[1]
            raw_height = row.height if row.height == row.height and row.height else None

            landmark_tag = None
            if bld_cls == 'chapel':
                landmark_tag = 'chapel'
            if bld_cls in ('church', 'cathedral') and poly_area > 60:
                landmark_tag = 'church'
            if lm_cfg and lm_cfg['spec'].get('lm'):
                landmark_tag = lm_cfg['spec']['lm']

            addr_hits = address_tree.query(poly.buffer(1.5))
            addresses = sorted({address_labels[i] for i in addr_hits if poly.buffer(1.5).contains(address_points[i])})

            roof_shape = ROOF_SHAPE_MAP.get(row.roof_shape if isinstance(row.roof_shape, str) else '', None)
            if not landmark_tag and (bld_cls == 'apartments' or (floors and floors >= 5) or (raw_height and raw_height >= 14)):
                roof_shape = 'flat'

            is_tenement = is_panel = False
            if is_urban and not landmark_tag:
                adj_buf = poly.buffer(0.8)
                is_attached = sum(1 for i in bld_poly_tree.query(adj_buf) if all_bld_polys[i].intersects(adj_buf)) >= 2
                is_tall = (floors and floors >= 9) or (raw_height and raw_height >= 28)
                if is_attached and is_panel_block(adj_buf):
                    roof_shape = 'flat'
                    is_panel = True
                elif (
                    is_attached
                    and not is_tall
                    and poly_area < 1500
                    and is_compact_obb(poly)
                    and compute_building_density(centroid) >= 0.12
                    and bld_cls in ('', 'residential', 'apartments', 'house', 'civic', 'commercial', 'retail', 'hotel', 'office', 'yes', 'school')
                ):
                    is_tenement = True
                    orig_roof = ROOF_SHAPE_MAP.get(row.roof_shape if isinstance(row.roof_shape, str) else '', None)
                    roof_shape = orig_roof if orig_roof and orig_roof != 'flat' else ('gabled' if hash_to_unit_float(osm_id + 't') < 0.75 else 'hipped')

            if not landmark_tag and osm_id in lidar_heights and poly_area > 80 and lidar_heights[osm_id][0] >= 6 and lidar_heights[osm_id][1] - lidar_heights[osm_id][0] < 0.4:
                roof_shape = 'flat'
                if is_tenement or (lidar_heights[osm_id][0] >= 11 and bld_cls in ('', 'residential', 'apartments', 'yes')):
                    is_tenement = False
                    is_panel = True
            if not landmark_tag and osm_id in lidar_heights and poly_area >= 25 and lidar_heights[osm_id][0] >= 2.2 and lidar_heights[osm_id][1] - lidar_heights[osm_id][0] < 0.35:
                roof_shape = 'flat'
                is_tenement = False
            if landmark_tag == 'radnice':
                roof_shape = 'hipped'
            if is_urban and not landmark_tag and roof_shape not in (None, 'flat') and poly_area > 2500:
                roof_shape = 'flat'
            if bld_cls == 'roof':
                roof_shape = 'flat'

            sat_override = satellite_data['b'].get(osm_id, {})
            lm_spec = lm_cfg['spec'] if lm_cfg else None
            if lm_spec:
                landmark_tag = 'dom'
                is_tenement = is_panel = False
                if 'r' in lm_spec:
                    roof_shape = 'flat' if lm_spec['r'] == 'flat' else lm_spec['r']

            if not landmark_tag and sat_override.get('rs') == 'flat':
                roof_shape = 'flat'
                is_tenement = False
            if (
                not landmark_tag
                and not is_panel
                and sat_override.get('rs') == 'pitched'
                and roof_shape in (None, 'flat')
                and poly_area < 700
                and bld_cls not in FLAT_CLASSES
                and not (floors and floors >= 6)
                and not (raw_height and raw_height >= 18)
            ):
                roof_shape = 'hipped' if hash_to_unit_float(osm_id + 't') < 0.6 else 'gabled'

            if roof_shape is None:
                if bld_cls in FLAT_CLASSES or poly_area > 650:
                    roof_shape = 'flat'
                elif bld_cls in PITCHED_CLASSES or (not bld_cls and 25 < poly_area < 400):
                    roof_shape = 'gabled' if hash_to_unit_float(osm_id) < 0.62 else 'hipped'
                else:
                    roof_shape = 'flat'

            if not landmark_tag and sat_override.get('r'):
                roof_shape = sat_override['r']
                is_tenement = is_panel = False

            obb = poly.minimum_rotated_rectangle
            obb_corners = list(obb.exterior.coords)[:4]
            edge1 = np.subtract(obb_corners[1], obb_corners[0])
            edge2 = np.subtract(obb_corners[2], obb_corners[1])
            if np.linalg.norm(edge1) < np.linalg.norm(edge2):
                edge1, edge2 = edge2, edge1
            obb_len, obb_width = float(np.linalg.norm(edge1)), float(np.linalg.norm(edge2))
            ridge_angle = math.atan2(edge1[1], edge1[0])
            if roof_ridges.get(osm_id) == 'S':
                ridge_angle += math.pi / 2
                obb_len, obb_width = obb_width, obb_len

            roof_sections = []
            if roof_shape != 'flat':
                if poly_area / max(obb.area, 1) > 0.9:
                    roof_sections = [[round(obb.centroid.x, 2), round(obb.centroid.y, 2), round(ridge_angle, 4), round(obb_len, 2), round(obb_width, 2)]]
                else:
                    roof_sections = decompose_roof_sections(poly, ridge_angle)
                if not roof_sections:
                    roof_shape = 'flat'

            roof_height = min(max(sec[4] for sec in roof_sections) / 2 * math.tan(math.radians(40)), 7.0) if roof_sections else 0.0
            if is_tenement:
                roof_height = min(roof_height, 6.5)
                total_height = (raw_height if raw_height and raw_height > 9 else (floors or (2 if poly_area < 130 else 4)) * 3.2 + 0.6) + roof_height * 0.6
            elif raw_height:
                total_height = raw_height
            elif floors:
                total_height = floors * 2.9 + 0.4 + roof_height * 0.8
            else:
                DEFAULT_CLASS_HEIGHTS = {
                    'garage': 2.6, 'garages': 2.6, 'shed': 2.4, 'carport': 2.4, 'apartments': 16,
                    'industrial': 7, 'warehouse': 7, 'school': 12, 'university': 14,
                    'civic': 9, 'commercial': 9, 'retail': 6,
                }
                total_height = DEFAULT_CLASS_HEIGHTS.get(bld_cls)
                if total_height is None:
                    total_height = 2.8 if poly_area < 30 else (6.0 + roof_height) if poly_area < 300 else 9.0

            if lm_spec and 'rh' in lm_spec and roof_sections:
                roof_height = float(lm_spec['rh'])
            eave_height = max(2.2, total_height - roof_height) if roof_shape != 'flat' else max(2.2, total_height)
            if lm_spec and 'h' in lm_spec:
                eave_height = float(lm_spec['h'])

            ext_xs, ext_ys = np.array(poly.exterior.coords).T
            ground_zs = get_ground_elevation(ext_xs, ext_ys)
            base_z, ground_z = float(ground_zs.min()) - 0.6, float(ground_zs.mean())

            if bld_cls in ('garage', 'garages', 'carport', 'shed'):
                facade_style = 2
            elif bld_cls == 'apartments' or (floors and floors >= 4) or total_height > 11:
                facade_style = 1
            elif bld_cls in ('industrial', 'warehouse', 'commercial', 'retail'):
                facade_style = 3
            else:
                facade_style = 0

            if is_tenement:
                facade_style = 11
            if is_panel:
                facade_style = 1

            if osm_id in lidar_heights and not lm_spec and not landmark_tag and bld_cls not in ('garage', 'garages', 'carport', 'shed', 'roof'):
                lidar_med, lidar_top = lidar_heights[osm_id]
                if is_panel:
                    lidar_med = get_connected_lidar_median(poly.buffer(0.8), lidar_med)
                if roof_shape == 'flat':
                    total_height = eave_height = lidar_med
                else:
                    total_height = max(lidar_top, lidar_med + 1)
                    eave_height = max(2.2, total_height - roof_height)

            if bld_cls == 'roof':
                facade_style = 12
                total_height = raw_height if raw_height and raw_height < 12 else 4.5
                eave_height = total_height
            if bld_cls in ('retail', 'commercial') and poly_area > 1500 and not is_tenement:
                facade_style = 3

            wall_palette = TENEMENT_WALL_PALETTE if facade_style == 11 else PANEL_WALL_PALETTE if facade_style == 1 else STANDARD_WALL_PALETTE
            color_seed_key = get_connected_group_key(poly.buffer(0.8)) if is_panel else osm_id
            wall_color = parse_color_rgb(row.facade_color, wall_palette[int(hash_to_unit_float(color_seed_key + 'w') * 1000) % len(wall_palette)])
            if facade_style == 2:
                wall_color = (196, 190, 178)

            roof_palette = FLAT_ROOF_PALETTE if roof_shape == 'flat' else TENEMENT_ROOF_PALETTE if is_tenement else PITCHED_ROOF_PALETTE
            roof_color = parse_color_rgb(row.roof_color, roof_palette[int(hash_to_unit_float(osm_id + 'r') * 1000) % len(roof_palette)])

            if 'rc' in sat_override and not landmark_tag:
                roof_color = tuple(sat_override['rc'])
            if 'wc' in sat_override and not landmark_tag:
                wall_color = tuple(sat_override['wc'])
            if lm_spec:
                if 'wc' in lm_spec:
                    wall_color = tuple(lm_spec['wc'])
                if 'rc' in lm_spec:
                    roof_color = tuple(lm_spec['rc'])
                if 'win' in lm_spec:
                    facade_style = lm_spec['win']

            footprint_ring = [[round(x, 2), round(y, 2)] for x, y in list(poly.exterior.coords)[:-1]]
            bld_obj = {
                'p': footprint_ring,
                'b': round(base_z, 2),
                'g': round(ground_z, 2),
                'e': round(ground_z + eave_height, 2),
                'r': roof_shape,
                'rh': round(roof_height, 2),
                'parts': roof_sections,
                'wc': wall_color,
                'rc': roof_color,
                's': facade_style,
                'id': bld_idx,
            }
            if landmark_tag:
                bld_obj['lm'] = landmark_tag

            if lm_spec:
                bld_obj['dn'] = lm_cfg['name']
                if lm_spec.get('canopy'):
                    bld_obj['cn'] = lm_spec['canopy']
                for blk in lm_spec.get('blocks', []):
                    obb = poly.minimum_rotated_rectangle
                    cs = list(obb.exterior.coords)[:3]
                    e1, e2 = np.subtract(cs[1], cs[0]), np.subtract(cs[2], cs[1])
                    if np.linalg.norm(e1) < np.linalg.norm(e2):
                        e1, e2 = e2, e1
                    blk_l, blk_w = float(np.linalg.norm(e1)) * blk['l'], float(np.linalg.norm(e2)) * blk['w']
                    bld_obj.setdefault('blk', []).append([
                        round(poly.centroid.x, 2), round(poly.centroid.y, 2), round(math.atan2(e1[1], e1[0]), 4),
                        round(blk_l, 2), round(blk_w, 2), round(ground_z + blk['h'], 2), list(blk.get('rc', roof_color)),
                    ])
                for tw in (lm_spec.get('towers', []) if piece_idx == tower_piece_idx else []):
                    tx, ty, tang = compute_landmark_feature_pos(pieces[piece_idx], tw['at'], lm_spec, lm_cfg, tw['w'], tw.get('side', 0.0), tw.get('t', 0.5))
                    if not is_in_core and not chunk_box.contains(Point(tx, ty)) and not extended_area.contains(Point(tx, ty)):
                        continue
                    bld_obj.setdefault('tw', []).append({
                        'x': round(tx, 2), 'y': round(ty, 2), 'a': round(tang, 4), 'w': tw['w'],
                        'h': round(ground_z + tw['h'], 2), 'to': round(ground_z + tw.get('to', tw['h']), 2),
                        'top': tw['top'], 'tc': tw.get('tc', [90, 96, 104]), 'wc': list(tw.get('wc', wall_color)),
                        'clock': tw.get('clock', 0), 'round': tw.get('round', 0), 'stone': tw.get('stone', 0),
                        'ruin': tw.get('ruin', 0), 'b': round(base_z, 2),
                    })
                if lm_spec.get('skylights') and piece_idx == tower_piece_idx:
                    inset = poly.buffer(-6.0)
                    if not inset.is_empty:
                        minx, miny, maxx, maxy = inset.bounds
                        bld_obj['rt'] = [
                            [round(gx, 2), round(gy, 2), 1, round(ridge_angle, 3)]
                            for gx in np.arange(minx, maxx, 13)
                            for gy in np.arange(miny, maxy, 11)
                            if inset.contains(Point(gx, gy)) and hash_to_unit_float(f'sk{gx:.0f}{gy:.0f}') < 0.45
                        ]
                for ch in (lm_spec.get('chimneys', []) if piece_idx == tower_piece_idx else []):
                    cx, cy, _ = compute_landmark_feature_pos(poly, ch['at'], lm_spec, lm_cfg, ch['r'] * 2, ch.get('side', 0.0))
                    bld_obj.setdefault('chs', []).append([
                        round(cx, 2), round(cy, 2), round(base_z, 2), round(ground_z + ch['h'], 2), ch['r'], list(ch['col']),
                    ])

            if landmark_tag == 'church' and 'Antonín' not in (bld_name or ''):
                tower_w = min(31.0, 10 + obb_len * 0.5)
                bld_obj['cht'] = [round(min(11.0, 5 + obb_len * 0.15), 1), round(tower_w, 1), round(tower_w + min(16.0, 4 + obb_len * 0.3), 1)]

            if is_urban and roof_shape == 'flat' and facade_style != 2 and poly_area > 300 and not landmark_tag:
                inset = poly.buffer(-2.0)
                if not inset.is_empty:
                    roof_rng = random.Random(osm_id)
                    minx, miny, maxx, maxy = inset.bounds
                    rt_items = []
                    for _ in range(int(min(90, poly_area / 60))):
                        rx, ry = roof_rng.uniform(minx, maxx), roof_rng.uniform(miny, maxy)
                        if inset.contains(Point(rx, ry)):
                            rt_items.append([round(rx, 2), round(ry, 2), roof_rng.choice((0, 0, 1, 1, 2)), round(ridge_angle, 3)])
                    bld_obj['rt'] = rt_items

            if landmark_tag == 'radnice' and town_squares:
                sq_center = town_squares[0].centroid
                nearest_wall_pt = shapely.ops.nearest_points(poly.exterior, sq_center)[0]
                wall_proj = poly.exterior.project(nearest_wall_pt)
                pt_prev = poly.exterior.interpolate(wall_proj - 2)
                pt_next = poly.exterior.interpolate(wall_proj + 2)
                bld_obj['rad'] = [
                    round(nearest_wall_pt.x, 2), round(nearest_wall_pt.y, 2),
                    round(math.atan2(pt_next.y - pt_prev.y, pt_next.x - pt_prev.x), 4),
                    round(math.atan2(sq_center.y - nearest_wall_pt.y, sq_center.x - nearest_wall_pt.x), 4),
                ]
                bld_obj['wc'] = (226, 212, 176)
                bld_obj['rc'] = (58, 66, 72)
                bld_obj['e'] = round(ground_z + 17.5, 2)

            if roof_sections and facade_style in (0, 1, 11) and not landmark_tag:
                sec_cx, sec_cy, sec_ang, sec_len, sec_wid = max(roof_sections, key=lambda q: q[3] * q[4])
                offset_s = (hash_to_unit_float(osm_id + 'c') - 0.5) * sec_len * 0.6
                offset_t = (0.12 + 0.18 * hash_to_unit_float(osm_id + 'd')) * sec_wid * (1 if hash_to_unit_float(osm_id + 'e') > 0.5 else -1)
                chimney_x = sec_cx + math.cos(sec_ang) * offset_s - math.sin(sec_ang) * offset_t
                chimney_y = sec_cy + math.sin(sec_ang) * offset_s + math.cos(sec_ang) * offset_t
                ridge_top_z = ground_z + eave_height + roof_height
                chimney_top_z = ridge_top_z - abs(offset_t) / (sec_wid / 2 + 0.45) * roof_height + 1.1
                bld_obj['ch'] = [round(chimney_x, 2), round(chimney_y, 2), round(chimney_top_z, 2)]
                if is_in_core:
                    residential_chimneys.append([chimney_x, chimney_y, chimney_top_z, bld_idx])
                if roof_shape == 'gabled' and sec_len > 9 and sec_wid > 6.5 and hash_to_unit_float(osm_id + 'v') < 0.55:
                    bld_obj['dm'] = 1

            if lm_spec:
                zone = zone_list[piece_idx]
                z_spec = zone or {}
                eff_ground = ground_z if shared_lm_ground is None else shared_lm_ground
                bld_obj['e'] = round(eff_ground + z_spec.get('h', bld_obj['e'] - ground_z), 2)
                bld_obj['g'] = round(eff_ground, 2)
                if 'rc' in z_spec:
                    bld_obj['rc'] = tuple(z_spec['rc'])
                if 'wc' in z_spec:
                    bld_obj['wc'] = tuple(z_spec['wc'])
                if 'win' in z_spec:
                    bld_obj['s'] = z_spec['win']
                if zone:
                    bld_obj['r'] = 'flat'
                    bld_obj['parts'] = []
                    bld_obj['rh'] = 0.0
                    bld_obj.pop('rt', None)
                deco_cfg = z_spec.get('deco') if zone else lm_spec.get('deco')
                if deco_cfg:
                    bld_obj['dz'] = generate_roof_decorations(poly, deco_cfg, bld_obj['e'], osm_id + str(piece_idx))

            if not is_in_core:
                bld_obj['gh'] = 1
            scene_buildings.append(bld_obj)
            building_mask_draw.polygon(to_tex_coords(poly.exterior.coords), fill=255)

            if not is_in_core:
                continue
            lon, lat = geo.convert_local_meters_to_lonlat(poly.centroid.x, poly.centroid.y)
            meta_top = ground_z + eave_height + roof_height
            if lm_spec:
                meta_top = bld_obj['e'] + (bld_obj['rh'] if bld_obj['r'] != 'flat' else 0.0)
                if bld_obj.get('dz'):
                    meta_top = max(meta_top, bld_obj['e'] + max(q[6] for q in bld_obj['dz']))
            building_metadata.append({
                'id': bld_idx, 'oid': osm_id, 'name': bld_name,
                'cls': CZECH_BUILDING_LABELS.get(bld_cls, 'budova' if not bld_cls else bld_cls),
                'floors': floors, 'h': round(total_height, 1), 'roof': roof_shape, 'addr': addresses,
                'lon': round(float(lon), 6), 'lat': round(float(lat), 6),
                'ring': footprint_ring, 'top': round(meta_top, 2), 'base': round(base_z, 2),
                **({'roof_geometry': {'eave': bld_obj['e'], 'height': bld_obj['rh'], 'parts': bld_obj['parts']}}
                   if not landmark_tag and roof_shape in ('gabled', 'hipped') and bld_obj['parts'] else {}),
            })

    # Standalone Tower Landmarks
    print("Standalone Tower Landmarks")
    for lm_cfg in landmarks:
        spec = lm_cfg['spec']
        if not spec.get('only'):
            continue
        lx, ly = lm_cfg['xy']
        if not extended_area_prep.contains(Point(lx, ly)):
            continue
        is_in_core = chunk_box.contains(Point(lx, ly))
        ground_z = float(get_ground_elevation(lx, ly))
        bld_idx = len(scene_buildings)
        half_w = spec['towers'][0]['w'] / 2
        ring = [[round(lx - half_w, 2), round(ly - half_w, 2)], [round(lx + half_w, 2), round(ly - half_w, 2)],
                [round(lx + half_w, 2), round(ly + half_w, 2)], [round(lx - half_w, 2), round(ly + half_w, 2)]]
        bld_obj = {
            'p': ring, 'b': round(ground_z - 0.6, 2), 'g': round(ground_z, 2), 'e': round(ground_z, 2),
            'r': 'flat', 'rh': 0.0, 'parts': [], 'wc': (150, 146, 136), 'rc': (150, 146, 136),
            's': 4, 'id': bld_idx, 'lm': spec.get('lm', 'dom'), 'dn': lm_cfg['name'], 'only': 1, 'tw': [],
        }
        for tw in spec['towers']:
            bld_obj['tw'].append({
                'x': round(lx, 2), 'y': round(ly, 2), 'a': 0.0, 'w': tw['w'],
                'h': round(ground_z + tw['h'], 2), 'to': round(ground_z + tw.get('to', tw['h']), 2),
                'top': tw['top'], 'tc': tw.get('tc', [90, 96, 104]), 'wc': [150, 146, 136],
                'clock': 0, 'round': tw.get('round', 0), 'stone': tw.get('stone', 0),
                'ruin': tw.get('ruin', 0), 'b': round(ground_z - 0.6, 2),
            })
        if not is_in_core:
            bld_obj['gh'] = 1
        scene_buildings.append(bld_obj)
        excl_r = 14 + spec['towers'][0]['w']
        tree_block_draw.ellipse([*to_tex_coords([(lx - excl_r, ly + excl_r)])[0], *to_tex_coords([(lx + excl_r, ly - excl_r)])[0]], fill=255)
        if is_in_core:
            tower_h = spec['towers'][0].get('to', spec['towers'][0]['h'])
            building_metadata.append({
                'id': bld_idx, 'oid': lm_cfg['name'], 'name': lm_cfg['name'], 'cls': spec.get('cls', 'dominanta'),
                'floors': None, 'h': round(float(tower_h), 1), 'roof': 'cone', 'addr': [],
                'lon': round(float(lm_cfg['lon']), 6), 'lat': round(float(lm_cfg['lat']), 6),
                'ring': ring, 'top': round(ground_z + tower_h, 2), 'base': round(ground_z - 0.6, 2),
            })

    # Industrial Chimneys, Masts & Towers
    print("Industrial Chimneys, Masts & Towers")
    chimney_records = json.load(open('assets/kominy.json')) if os.path.exists('assets/kominy.json') else []
    for chim in chimney_records:
        cx, cy = geo.convert_lonlat_to_local_meters(chim['lon'], chim['lat'])
        if not extended_area_prep.contains(Point(cx, cy)):
            continue
        is_in_core = chunk_box.contains(Point(cx, cy))
        ground_z = float(get_ground_elevation(cx, cy))
        chim_h = float(chim['h'])
        override = next((q for q in manual_overrides.get('kominy', []) if math.dist((cx, cy), geo.convert_lonlat_to_local_meters(q['lon'], q['lat'])) < 20), {})
        kind_code = {'vez': 'v', 'stozar': 'm', 'nadrz': 't', 'rozhledna': 'r'}.get(chim['k'], 'c')
        radius_m = chim.get('r') or {'v': 10.0, 'm': 2.5, 't': 5.0, 'r': 3.2, 'c': max(2.5, chim_h * 0.045)}[kind_code]
        default_cols = {'v': [176, 172, 160], 'm': [150, 152, 156], 't': [186, 184, 176], 'r': [176, 112, 62]}
        wall_col = list(override.get('wc', default_cols.get(kind_code, [168, 162, 150])))
        bld_obj = {
            'p': [[round(cx - 0.1, 2), round(cy - 0.1, 2)], [round(cx + 0.1, 2), round(cy - 0.1, 2)], [round(cx + 0.1, 2), round(cy + 0.1, 2)]],
            'b': round(ground_z - 0.6, 2), 'g': round(ground_z, 2), 'e': round(ground_z + chim_h, 2),
            'r': 'flat', 'rh': 0.0, 'parts': [], 'wc': tuple(wall_col), 'rc': tuple(wall_col),
            's': 4, 'id': len(scene_buildings), 'lm': 'dom', 'dn': 'komín', 'only': 1, 'tw': [],
            'stk': {
                'k': kind_code, 'x': round(cx, 2), 'y': round(cy, 2), 'r': round(radius_m, 2), 'wc': wall_col,
                'pr': override.get('pr', 1 if chim_h >= 100 and kind_code == 'c' else 0),
            },
        }
        if not is_in_core:
            bld_obj['gh'] = 1
        elif kind_code == 'c':
            residential_chimneys.append([cx, cy, ground_z + chim_h, bld_obj['id']])
        scene_buildings.append(bld_obj)
        excl_r = radius_m + 3
        tree_block_draw.ellipse([*to_tex_coords([(cx - excl_r, cy + excl_r)])[0], *to_tex_coords([(cx + excl_r, cy - excl_r)])[0]], fill=255)

    # Dams
    print("Dams")
    for _, row in infra_df.iterrows():
        if row['class'] != 'dam' or row.geom.geom_type not in ('Polygon', 'MultiPolygon'):
            continue
        for poly in extract_polygons(row.geom):
            if not extended_area.intersects(poly) or poly.area < 20:
                continue
            poly = shapely.geometry.polygon.orient(poly.simplify(0.3), 1.0)
            ext_xs, ext_ys = np.array(poly.exterior.coords).T
            zs = get_ground_elevation(ext_xs, ext_ys)
            is_in_core = chunk_box.contains(poly.centroid)
            bld_obj = {
                'p': [[round(x, 2), round(y, 2)] for x, y in list(poly.exterior.coords)[:-1]],
                'b': round(float(zs.min()) - 0.6, 2), 'g': round(float(zs.mean()), 2), 'e': round(float(zs.max()) + 1.2, 2),
                'r': 'flat', 'rh': 0.0, 'parts': [], 'wc': (150, 146, 136), 'rc': (172, 168, 158), 's': 5, 'id': len(scene_buildings),
            }
            if not is_in_core:
                bld_obj['gh'] = 1
            for lm_cfg in landmarks:
                if 'towers' in lm_cfg['spec'] and not lm_cfg['spec'].get('only') and poly.buffer(30).contains(Point(*lm_cfg['xy'])) and row.id not in landmarks_by_bld_id:
                    bld_obj['lm'] = 'dom'
                    bld_obj['dn'] = lm_cfg['name']
                    bld_obj['tw'] = []
                    for tw in lm_cfg['spec']['towers']:
                        tx, ty, tang = compute_landmark_feature_pos(poly, tw['at'], lm_cfg['spec'], lm_cfg, tw['w'], tw.get('side', 0.0), tw.get('t', 0.5))
                        bld_obj['tw'].append({
                            'x': round(tx, 2), 'y': round(ty, 2), 'a': round(tang, 4), 'w': tw['w'],
                            'h': round(bld_obj['e'] + tw['h'], 2), 'to': round(bld_obj['e'] + tw.get('to', tw['h']), 2),
                            'top': tw['top'], 'tc': tw.get('tc', [90, 96, 104]), 'wc': [150, 146, 136],
                            'clock': 0, 'round': tw.get('round', 0), 'stone': 1, 'ruin': 0, 'b': round(bld_obj['e'] - 0.5, 2),
                        })
            scene_buildings.append(bld_obj)
            building_mask_draw.polygon(to_tex_coords(poly.exterior.coords), fill=255)

    # Barracks HQ Special Handling
    print("Barracks HQ Special Handling")
    barracks_polys = [row.geom for _, row in land_use_df.iterrows() if row['class'] == 'barracks' and 'Horní' in (get_primary_name(row) or '')]
    if barracks_polys:
        gates = sorted(
            [row.geom for _, row in infra_df.iterrows() if row['class'] == 'gate' and row.geom.geom_type == 'Point' and barracks_polys[0].buffer(5).contains(row.geom)],
            key=lambda g: g.x,
        )[:2]
        if gates:
            gate_x, gate_y = sum(g.x for g in gates) / len(gates), sum(g.y for g in gates) / len(gates)
            candidates = [b for b in scene_buildings if barracks_polys[0].contains(Polygon(b['p']).centroid) and Polygon(b['p']).area > 300]
            if candidates:
                hq_bld = min(candidates, key=lambda b: Polygon(b['p']).centroid.distance(Point(gate_x, gate_y)))
                hq_bld['lm'] = 'hq'
                hq_bld.pop('ch', None)
                hq_bld.pop('dm', None)
                if not hq_bld['parts']:
                    hq_poly = Polygon(hq_bld['p'])
                    obb = hq_poly.minimum_rotated_rectangle
                    cs = list(obb.exterior.coords)[:4]
                    e1, e2 = np.subtract(cs[1], cs[0]), np.subtract(cs[2], cs[1])
                    if np.linalg.norm(e1) < np.linalg.norm(e2):
                        e1, e2 = e2, e1
                    hq_bld['parts'] = [[round(obb.centroid.x, 2), round(obb.centroid.y, 2), round(math.atan2(e1[1], e1[0]), 4), round(float(np.linalg.norm(e1)), 2), round(float(np.linalg.norm(e2)), 2)]]
                    hq_bld['r'] = 'hipped'
                    hq_bld['rh'] = round(min(np.linalg.norm(e2) / 2 * math.tan(math.radians(40)), 7), 2)
                    hq_bld['e'] = round(hq_bld['e'] - 0.0, 2)
                print('velitelství kasáren', hq_bld['id'])

    print('budov', len(scene_buildings), 'z toho od sousedů', sum(1 for b in scene_buildings if b.get('gh')), 'komínů', len(residential_chimneys))

    # ---------------- Street Furniture & Props ----------------
    print("Street Furniture & Props")
    street_props = {'bench': [], 'lamp': [], 'bin': [], 'recycling': [], 'shelter': [], 'tower': [], 'pole': []}
    existing_lamp_pts = []

    for _, row in infra_df.iterrows():
        pt = row.geom
        prop_cls = row['class']
        if pt.geom_type != 'Point' or not chunk_box.contains(pt):
            continue
        ground_z = float(get_ground_elevation(pt.x, pt.y))
        nearest_road = find_nearest_road_direction(pt, non_rail_tree, non_rail_lines, 15)
        yaw = nearest_road[0] if nearest_road else hash_to_unit_float(world_coord_key(pt.x, pt.y, center_x, center_y)) * 6.28
        if prop_cls == 'bench':
            street_props['bench'].append([pt.x, pt.y, ground_z, yaw])
        elif prop_cls == 'street_lamp':
            street_props['lamp'].append([pt.x, pt.y, ground_z, yaw])
            existing_lamp_pts.append(pt)
        elif prop_cls in ('waste_basket', 'waste_disposal'):
            street_props['bin'].append([pt.x, pt.y, ground_z, yaw])
        elif prop_cls == 'recycling':
            street_props['recycling'].append([pt.x, pt.y, ground_z, yaw])
        elif prop_cls == 'power_tower':
            street_props['tower'].append([pt.x, pt.y, ground_z, yaw])
        elif prop_cls == 'power_pole':
            street_props['pole'].append([pt.x, pt.y, ground_z, yaw])
        elif prop_cls in ('bus_stop', 'platform'):
            if any(math.hypot(pt.x - s[0], pt.y - s[1]) < 14 for s in street_props['shelter']):
                continue
            nearest_drive = find_nearest_road_direction(pt, drivable_tree, drivable_lines, 20)
            street_props['shelter'].append([pt.x, pt.y, ground_z, nearest_drive[0] if nearest_drive else yaw])

    lamp_tree = shapely.STRtree(existing_lamp_pts) if existing_lamp_pts else None
    bld_mask_arr = np.asarray(building_mask_img)

    for line, width, r_cls, _ in drivable_lines:
        if r_cls in ('service', 'motorway'):
            continue
        cursor, side = 12.0, 1
        while cursor < line.length - 5:
            pt = line.interpolate(cursor)
            pt_prev = line.interpolate(max(0, cursor - 1))
            pt_next = line.interpolate(min(line.length, cursor + 1))
            heading = math.atan2(pt_next.y - pt_prev.y, pt_next.x - pt_prev.x)
            offset = (width / 2 + 1.3) * side
            lamp_x = pt.x + math.cos(heading + math.pi / 2) * offset
            lamp_y = pt.y + math.sin(heading + math.pi / 2) * offset
            px, py = to_tex_pixel(lamp_x, lamp_y)
            is_valid_spot = (
                chunk_box.contains(Point(lamp_x, lamp_y))
                and 0 <= px < ext_tex_width
                and 0 <= py < ext_tex_width
                and bld_mask_arr[py, px] == 0
            )
            if is_valid_spot and lamp_tree is not None:
                nearest_idx = lamp_tree.nearest(Point(lamp_x, lamp_y))
                is_valid_spot = existing_lamp_pts[nearest_idx].distance(Point(lamp_x, lamp_y)) > 18
            if is_valid_spot and not any(abs(lamp_x - s[0]) < 12 and abs(lamp_y - s[1]) < 12 for s in street_props['lamp'][-6:]):
                street_props['lamp'].append([lamp_x, lamp_y, float(get_ground_elevation(lamp_x, lamp_y)), heading])
            cursor += 34
            side = -side

    # Residential Wheelie Bins
    street_props['wheelie'] = []
    residential_roads = [(l, w) for l, w, c, _ in drivable_lines if c in ('residential', 'unclassified', 'living_street', 'tertiary')]
    residential_road_tree = shapely.STRtree([item[0] for item in residential_roads])

    for bld in scene_buildings:
        if bld['s'] != 0 or bld.get('lm') or bld.get('gh'):
            continue
        bld_poly = Polygon(bld['p'])
        centroid = bld_poly.centroid
        c_key = world_coord_key(centroid.x, centroid.y, center_x, center_y)
        if bld_poly.area < 50 or hash_to_unit_float('bin' + c_key) > 0.55:
            continue
        bin_rng = random.Random('bin' + c_key)
        road_idx = residential_road_tree.nearest(centroid)
        road_line, road_w = residential_roads[road_idx]
        if road_line.distance(centroid) > 28:
            continue
        curb_pt = road_line.interpolate(road_line.project(centroid))
        dx, dy = centroid.x - curb_pt.x, centroid.y - curb_pt.y
        dist = math.hypot(dx, dy) or 1.0
        dx, dy = dx / dist, dy / dist
        pt_prev = road_line.interpolate(max(0, road_line.project(curb_pt) - 1))
        pt_next = road_line.interpolate(min(road_line.length, road_line.project(curb_pt) + 1))
        heading = math.atan2(pt_next.y - pt_prev.y, pt_next.x - pt_prev.x)
        for k in range(bin_rng.choice((1, 2, 2, 3))):
            lateral = (k - 0.5) * 0.75
            bx = curb_pt.x + dx * (road_w / 2 + 2.4) + math.cos(heading) * lateral
            by = curb_pt.y + dy * (road_w / 2 + 2.4) + math.sin(heading) * lateral
            if chunk_box.contains(Point(bx, by)):
                bin_color = (0, 0, 1, 2)[k % 4] if k else bin_rng.choice((0, 0, 3))
                street_props['wheelie'].append([bx, by, float(get_ground_elevation(bx, by)), math.atan2(dy, dx), bin_color])

    # Linear Features (Fences, Hedges, Walls, Powerlines, Ski Lifts)
    linear_features = {'fence': [], 'hedge': [], 'wall': [], 'power': [], 'lift': []}
    LINEAR_CLASS_MAP = {
        'fence': 'fence', 'hedge': 'hedge', 'wall': 'wall', 'retaining_wall': 'wall',
        'power_line': 'power', 'minor_line': 'power',
        'chair_lift': 'lift', 'drag_lift': 'lift', 't-bar': 'lift', 'platter': 'lift', 'cable_car': 'lift', 'gondola': 'lift',
    }
    for _, row in infra_df.iterrows():
        target_key = LINEAR_CLASS_MAP.get(row['class'])
        if not target_key:
            continue
        for line in extract_linestrings(row.geom.intersection(chunk_box)):
            dense_line = line.segmentize(4.0)
            xs, ys = np.array(dense_line.coords).T
            linear_features[target_key].append([
                [round(x, 2), round(y, 2), round(float(z), 2)] for x, y, z in zip(xs, ys, get_ground_elevation(xs, ys))
            ])

    for key in street_props:
        street_props[key] = [[round(val, 3) for val in item] for item in street_props[key]]

    parked_cars = [[round(x, 2), round(y, 2), round(float(get_ground_elevation(x, y)), 2), round(a, 3), ci] for x, y, a, ci in parked_cars]
    parked_cars += [[round(x, 2), round(y, 2), round(z, 2), round(a, 3), ci] for x, y, z, a, ci in rooftop_cars if chunk_box.contains(Point(x, y))]

    parked_buses = []
    for start_ll, end_ll, bus_kind in manual_overrides['buses']:
        x1, y1 = geo.convert_lonlat_to_local_meters(start_ll[1], start_ll[0])
        x2, y2 = geo.convert_lonlat_to_local_meters(end_ll[1], end_ll[0])
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        if chunk_box.contains(Point(bx, by)):
            parked_buses.append([round(bx, 2), round(by, 2), round(float(get_ground_elevation(bx, by)), 2), round(math.atan2(y2 - y1, x2 - x1), 3), bus_kind])

    tunnel_portals = [[round(x, 2), round(y, 2), round(float(get_ground_elevation(x, y)), 2), round(a, 3)] for x, y, a in tunnel_portals if chunk_box.contains(Point(x, y))]

    if os.path.exists('assets/potrubi.json'):
        linear_features['pipe'] = []
        for pipe_coords in json.load(open('assets/potrubi.json')):
            for line in extract_linestrings(LineString([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in pipe_coords]).intersection(extended_area)):
                xy_pts = shapely.get_coordinates(shapely.segmentize(line, 10.0))
                linear_features['pipe'].append([
                    [round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                    for (x, y), z in zip(xy_pts, get_ground_elevation(xy_pts[:, 0], xy_pts[:, 1]))
                ])

    print({k: len(v) for k, v in street_props.items()}, 'aut', len(parked_cars), {k: len(v) for k, v in linear_features.items()})

    # ---------------- Tree Placement & Filtering ----------------
    print("Tree Placement & Filtering")
    tree_exclusion_mask = np.maximum(np.asarray(tree_block_img), np.asarray(building_mask_img.filter(ImageFilter.MaxFilter(13))))

    def is_tree_spot_free(x, y):
        px, py = to_tex_pixel(x, y)
        return 0 <= px < ext_tex_width and 0 <= py < ext_tex_width and tree_exclusion_mask[py, px] == 0

    def are_tree_spots_free_vec(xs, ys):
        pxs = ((xs + extended_half) / tex_meters_per_px).astype(int)
        pys = ((extended_half - ys) / tex_meters_per_px).astype(int)
        in_bounds = (pxs >= 0) & (pxs < ext_tex_width) & (pys >= 0) & (pys < ext_tex_width)
        valid = np.zeros(len(xs), bool)
        valid[in_bounds] = tree_exclusion_mask[pys[in_bounds], pxs[in_bounds]] == 0
        return valid

    raw_trees = []

    def scatter_trees_in_geometry(geometry, spacing_m, conifer_ratio, min_h, max_h, tree_kind=0):
        spacing_id = int(round(spacing_m * 10))
        for poly in extract_polygons(geometry.intersection(extended_area)):
            minx, miny, maxx, maxy = poly.bounds
            grid_kx = np.arange(math.floor((minx + center_x) / spacing_m), math.ceil((maxx + center_x) / spacing_m) + 1)
            grid_ky = np.arange(math.floor((miny + center_y) / spacing_m), math.ceil((maxy + center_y) / spacing_m) + 1)
            mesh_kx, mesh_ky = [arr.ravel() for arr in np.meshgrid(grid_kx, grid_ky)]
            cand_x = mesh_kx * spacing_m - center_x + (vectorized_int_hash(mesh_kx, mesh_ky, spacing_id, 1) - 0.5) * 0.9 * spacing_m
            cand_y = mesh_ky * spacing_m - center_y + (vectorized_int_hash(mesh_kx, mesh_ky, spacing_id, 2) - 0.5) * 0.9 * spacing_m
            valid_mask = shapely.contains_xy(poly, cand_x, cand_y)
            valid_mask[valid_mask] = are_tree_spots_free_vec(cand_x[valid_mask], cand_y[valid_mask])
            heights = min_h + vectorized_int_hash(mesh_kx, mesh_ky, spacing_id, 3) * (max_h - min_h)
            is_conifer = vectorized_int_hash(mesh_kx, mesh_ky, spacing_id, 4) < conifer_ratio
            for idx in np.nonzero(valid_mask)[0]:
                raw_trees.append((float(cand_x[idx]), float(cand_y[idx]), float(heights[idx]), 1 if is_conifer[idx] else 0, tree_kind))

    for _, row in land_cover_df.iterrows():
        lc_cls = row['class']
        geom = row.geom
        if lc_cls in ('forest', 'wood') and geom.geom_type != 'Point':
            scatter_trees_in_geometry(geom, 4.2, 0.55, 12, 24)
        elif lc_cls in ('scrub', 'shrub') and geom.geom_type != 'Point':
            scatter_trees_in_geometry(geom, 5, 0.1, 2, 4, 1)
        elif lc_cls == 'tree' and geom.geom_type == 'Point':
            if extended_area_prep.contains(geom):
                t_key = world_coord_key(geom.x, geom.y, center_x, center_y)
                raw_trees.append((geom.x, geom.y, 8 + hash_to_unit_float('t' + t_key) * 8, 0, 0))
        elif lc_cls == 'tree_row':
            for line in extract_linestrings(geom):
                cursor = 0
                while cursor < line.length:
                    pt = line.interpolate(cursor)
                    if extended_area_prep.contains(pt):
                        r_key = world_coord_key(pt.x, pt.y, center_x, center_y)
                        raw_trees.append((pt.x, pt.y, 8 + hash_to_unit_float('r' + r_key) * 6, 0, 0))
                    cursor += 7

    LAND_USE_TREE_SPACING = {
        'park': 11, 'cemetery': 9, 'residential': 16, 'garden': 12, 'grass': 20,
        'village_green': 18, 'recreation_ground': 20, 'allotments': 14, 'kindergarten': 14,
    }
    for _, row in land_use_df.iterrows():
        spacing = LAND_USE_TREE_SPACING.get(row['class']) or (LAND_USE_TREE_SPACING['park'] if row.id in green_plaza_ids else None)
        if spacing:
            scatter_trees_in_geometry(row.geom, spacing, 0.25, 6, 15)

    open_land_union = shapely.unary_union(
        [g for c, g in zip(land_use_df['class'], land_use_df.geom) if c in ('meadow', 'farmland', 'grass', 'village_green', 'downhill', 'pitch')]
        + [g for c, g in zip(land_cover_df['class'], land_cover_df.geom) if c in ('grassland', 'meadow') and g.geom_type != 'Point']
        + meadow_polys
        + arable_polys
    )
    sparse_tree_area = extended_area.difference(open_land_union.buffer(0)) if not open_land_union.is_empty else extended_area
    scatter_trees_in_geometry(sparse_tree_area, 26, 0.3, 6, 14)

    if forest_mask is not None:
        orchard_and_garden_prep = shapely.prepared.prep(
            shapely.unary_union(orchard_polys + [row.geom for _, row in land_use_df.iterrows() if row['class'] in ('residential', 'garden', 'allotments')]).buffer(0)
            .difference(shapely.unary_union(arable_polys + meadow_polys).buffer(0))
        )

        def passes_forest_mask(tree_tuple):
            px, py = to_tex_pixel(tree_tuple[0], tree_tuple[1])
            if not (0 <= px < ext_tex_width and 0 <= py < ext_tex_width) or forest_mask[py, px]:
                return True
            t_key = world_coord_key(tree_tuple[0], tree_tuple[1], center_x, center_y)
            return hash_to_unit_float('z' + t_key) < 0.3 and orchard_and_garden_prep.contains(Point(tree_tuple[0], tree_tuple[1]))

        raw_trees = [t for t in raw_trees if passes_forest_mask(t)]

    for area_cfg in area_overrides.values():
        if area_cfg.get('skryt'):
            continue
        zone_prep = shapely.prepared.prep(Polygon([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in area_cfg['poly']]))
        raw_trees = [t for t in raw_trees if not zone_prep.contains(Point(t[0], t[1]))]
        for lat, lon, tree_h, is_con in area_cfg['trees']:
            tx, ty = (float(v) for v in geo.convert_lonlat_to_local_meters(lon, lat))
            if extended_area_prep.contains(Point(tx, ty)) and is_tree_spot_free(tx, ty):
                raw_trees.insert(0, (tx, ty, tree_h, is_con, 0))

    # Deduplicate trees on a 3m world grid
    occupied_tree_cells, deduplicated_trees = set(), []
    for tree in raw_trees:
        cell_key = (round((tree[0] + center_x) / 3.0), round((tree[1] + center_y) / 3.0))
        if cell_key in occupied_tree_cells:
            continue
        occupied_tree_cells.add(cell_key)
        deduplicated_trees.append(tree)

    for area_cfg in area_overrides.values():
        if not area_cfg.get('skryt'):
            continue
        zone_prep = shapely.prepared.prep(Polygon([geo.convert_lonlat_to_local_meters(lon, lat) for lat, lon in area_cfg['poly']]))
        deduplicated_trees = [t[:4] + (2,) if zone_prep.contains(Point(t[0], t[1])) else t for t in deduplicated_trees]
        for lat, lon, tree_h, is_con in area_cfg['trees']:
            tx, ty = (float(v) for v in geo.convert_lonlat_to_local_meters(lon, lat))
            if extended_area_prep.contains(Point(tx, ty)):
                deduplicated_trees.append((tx, ty, tree_h, is_con, 0))

    # Remove trees overhanging roads/bridges
    clearance_roads = (
        [(line, width) for _, line, r_cls, width, _ in road_entries if r_cls in DRIVABLE_CLASSES and r_cls != 'service']
        + [(deck[0], deck[4]) for deck in bridge_decks if deck[6] in DRIVABLE_CLASSES and deck[6] != 'service']
    )
    if clearance_roads:
        road_clear_tree = shapely.STRtree([item[0] for item in clearance_roads])
        count_before = len(deduplicated_trees)

        def is_clear_of_roads(tree):
            canopy_r = tree[2] * (0.2 if tree[3] else 0.33)
            tree_pt = Point(tree[0], tree[1])
            for idx in road_clear_tree.query(tree_pt.buffer(canopy_r + 8)):
                r_line, r_width = clearance_roads[idx]
                if r_line.distance(tree_pt) < r_width / 2 + canopy_r:
                    return False
            return True

        deduplicated_trees = [t for t in deduplicated_trees if is_clear_of_roads(t)]
        print('stromů nad vozovkou odstraněno', count_before - len(deduplicated_trees))

    tree_elevations = get_ground_elevation(np.array([t[0] for t in deduplicated_trees]), np.array([t[1] for t in deduplicated_trees]))
    scene_trees = [
        [round(t[0], 2), round(t[1], 2), round(float(z), 2), round(t[2], 1), t[3], t[4]]
        for t, z in zip(deduplicated_trees, tree_elevations)
    ]
    print('stromů', len(scene_trees))

    # ---------------- Cemeteries, Graves & Statues ----------------
    print("Cemeteries, Graves & Statues")
    cem_cfg = manual_overrides.get('cemetery', {})
    cemetery_union = shapely.unary_union([row.geom for _, row in land_use_df.iterrows() if row['class'] == 'cemetery'])
    cemetery_prep = shapely.prepared.prep(cemetery_union)
    scene_trees = [
        t[:5] + [2]
        if cemetery_prep.contains(Point(t[0], t[1])) and hash_to_unit_float(world_coord_key(t[0], t[1], center_x, center_y)) >= cem_cfg.get('tree_keep', 0.3)
        else t
        for t in scene_trees
    ]
    no_grave_zone = shapely.geometry.GeometryCollection()
    run_custom_hook('cemetery')
    scene_trees = [t for t in scene_trees if extended_area_prep.contains(Point(t[0], t[1]))]
    print('skrytých stromů na hřbitovech', sum(t[5] == 2 for t in scene_trees))

    grave_rng = random.Random(7)
    graves, statues = [], []
    visible_tree_tree = shapely.STRtree([Point(t[0], t[1]) for t in scene_trees if t[5] != 2])
    grave_blockers = shapely.unary_union(
        [line.buffer(width / 2 + 0.9) for line, width, _, _ in non_rail_lines]
        + [Polygon(bld['p']).buffer(1.5) for bld in scene_buildings]
        + [line.buffer(1.2) for _, row in infra_df.iterrows() if row['class'] in ('wall', 'fence') for line in extract_linestrings(row.geom)]
    )

    for _, row in land_use_df.iterrows():
        if row['class'] != 'cemetery':
            continue
        for poly in extract_polygons(row.geom.intersection(chunk_box)):
            inset = poly.buffer(cem_cfg.get('margin', -1.5))
            if inset.is_empty:
                continue
            valid_grave_area = shapely.prepared.prep(inset.difference(grave_blockers).difference(no_grave_zone.buffer(1.0)))
            obb = poly.minimum_rotated_rectangle
            corners = [np.array(q) for q in list(obb.exterior.coords)[:4]]
            e1, e2 = corners[1] - corners[0], corners[3] - corners[0]
            if np.linalg.norm(e1) < np.linalg.norm(e2):
                e1, e2 = e2, e1
                corners = [corners[0], corners[3], corners[2], corners[1]]
            cem_l, cem_w = np.linalg.norm(e1), np.linalg.norm(e2)
            unit_u, unit_v = e1 / cem_l, e2 / cem_w
            grave_angle = math.atan2(unit_v[1], unit_v[0])
            is_jewish = 'Žid' in (get_primary_name(row) or '')
            row_offset = cem_cfg.get('row0', 1.0)
            while row_offset + 2.2 < cem_w:
                col_offset = cem_cfg.get('col0', 0.8)
                while col_offset + 1.0 < cem_l:
                    grave_center = corners[0] + unit_u * (col_offset + 0.5) + unit_v * (row_offset + 1.0)
                    grave_corners = [corners[0] + unit_u * (col_offset + du) + unit_v * (row_offset + dv) for du, dv in ((0, 0), (1.0, 0), (1.0, 2.0), (0, 2.0))]
                    if all(valid_grave_area.contains(Point(*pt)) for pt in grave_corners) and grave_rng.random() < cem_cfg.get('prob', 0.75 if is_jewish else 0.9):
                        if len(visible_tree_tree.query(Point(*grave_center).buffer(cem_cfg.get('tree_dist', 1.6)))) == 0:
                            if not is_jewish:
                                surface_draw.polygon(to_tex_coords([tuple(pt) for pt in grave_corners]), fill=SURFACE_IDS['grave'])
                            headstone_pt = corners[0] + unit_u * (col_offset + 0.5) + unit_v * (row_offset + 1.9)
                            graves.append([
                                round(float(headstone_pt[0]), 2), round(float(headstone_pt[1]), 2), round(float(get_ground_elevation(*headstone_pt)), 2),
                                round(grave_angle, 3), round(grave_rng.uniform(0.55, 1.15) if not is_jewish else grave_rng.uniform(0.5, 0.9), 2), grave_rng.randrange(4),
                            ])
                    col_offset += cem_cfg.get('dcol', 1.45)
                row_offset += cem_cfg.get('drow', 3.1)

    for _, row in infra_df.iterrows():
        pt = row.geom
        if row['class'] == 'artwork' and pt.geom_type == 'Point' and chunk_box.contains(pt):
            if any(r2['class'] == 'cemetery' and r2.geom.buffer(3).contains(pt) for _, r2 in land_use_df.iterrows()):
                statues.append([round(pt.x, 2), round(pt.y, 2), round(float(get_ground_elevation(pt.x, pt.y)), 2), 0.0])

    street_props['planter'] = street_props['slab'] = street_props['pavilion'] = street_props['shed'] = []
    street_props['grave'] = graves
    street_props['statue'] = statues
    street_props['wheelie'] = [[round(val, 3) for val in item] for item in street_props['wheelie']]
    run_custom_hook('props')
    print('hrobů', len(graves), 'soch', len(statues))

    # ---------------- Seasonal Ground Texture Generation ----------------
    print("Seasonal Ground Texture Generation")
    SEASON_PALETTES = {
        'summer': {
            'grass': (106, 160, 74), 'garden': (118, 168, 80), 'park': (104, 164, 72), 'meadow': (140, 172, 86),
            'forest': (66, 112, 56), 'scrub': (92, 128, 62), 'pitch': (80, 156, 72), 'track': (178, 92, 72),
            'sand': (214, 192, 138), 'paved': (196, 186, 160), 'industrial': (156, 152, 142), 'parking': (120, 120, 124),
            'road': (92, 92, 98), 'marking': (236, 232, 214), 'curb': (176, 172, 162), 'dirt': (170, 146, 102),
            'water': (64, 124, 190), 'pool': (92, 196, 230), 'stream': (72, 132, 196), 'rail': (88, 84, 82),
            'cemetery': (88, 132, 70), 'wetland': (100, 146, 92), 'rock': (150, 146, 136), 'construction': (164, 132, 96),
            'footway': (214, 206, 186), 'pool_edge': (236, 240, 240), 'farmland': (168, 172, 96), 'cobble': (200, 188, 164),
            'grave': (156, 152, 144), 'manhole': (66, 66, 70), 'court_blue': (70, 112, 170), 'zone': (255, 0, 220),
            'zone2': (0, 230, 230), 'piste': (150, 178, 92), 'skijump': (98, 160, 150),
            'wheat': (214, 186, 106), 'wheat2': (194, 162, 86), 'rape': (150, 148, 82), 'rape2': (128, 126, 70),
            'maize': (80, 134, 58), 'maize2': (58, 106, 44), 'soil': (110, 142, 66), 'soil2': (128, 104, 70),
            'plaza': (186, 186, 184), 'plaza2': (206, 206, 200), 'deck': (146, 104, 88),
        },
    }
    spring_pal = dict(SEASON_PALETTES['summer'])
    spring_pal.update({
        'grass': (122, 176, 80), 'garden': (132, 184, 86), 'park': (118, 180, 78), 'meadow': (150, 188, 90),
        'forest': (78, 128, 62), 'farmland': (132, 150, 90),
        'wheat': (104, 146, 72), 'wheat2': (88, 128, 62), 'rape': (238, 214, 60), 'rape2': (216, 194, 50),
        'maize': (138, 106, 74), 'maize2': (112, 86, 60), 'soil': (130, 98, 68), 'soil2': (106, 80, 56),
    })
    SEASON_PALETTES['spring'] = spring_pal

    autumn_pal = dict(SEASON_PALETTES['summer'])
    autumn_pal.update({
        'grass': (122, 154, 72), 'garden': (128, 158, 76), 'park': (120, 156, 70), 'meadow': (156, 160, 88),
        'forest': (92, 104, 58), 'scrub': (124, 116, 64), 'farmland': (176, 150, 96), 'cemetery': (104, 128, 70),
        'wheat': (204, 182, 126), 'wheat2': (164, 132, 88), 'rape': (112, 150, 74), 'rape2': (98, 134, 64),
        'maize': (152, 126, 86), 'maize2': (124, 98, 66), 'soil': (122, 92, 62), 'soil2': (98, 72, 50),
    })
    SEASON_PALETTES['autumn'] = autumn_pal

    winter_pal = dict(SEASON_PALETTES['summer'])
    snow_rgb = (234, 239, 246)
    winter_pal.update({
        k: snow_rgb for k in (
            'grass', 'garden', 'park', 'meadow', 'pitch', 'sand', 'cemetery', 'farmland',
            'construction', 'wetland', 'track', 'rock', 'piste', 'skijump', 'wheat', 'rape', 'maize', 'soil',
        )
    })
    winter_pal.update({k: (212, 218, 228) for k in ('wheat2', 'rape2', 'maize2', 'soil2')})
    winter_pal.update({
        'forest': (214, 222, 232), 'scrub': (220, 226, 234), 'paved': (222, 226, 232), 'industrial': (206, 210, 216),
        'parking': (176, 180, 188), 'road': (104, 106, 114), 'marking': (190, 192, 196), 'curb': (226, 230, 236),
        'dirt': (214, 214, 218), 'water': (186, 210, 228), 'pool': (150, 168, 184), 'stream': (58, 96, 136),
        'footway': (226, 230, 236), 'cobble': (206, 210, 218), 'pool_edge': (230, 234, 238), 'grave': (222, 226, 232),
        'manhole': (90, 92, 98), 'plaza': (208, 212, 220), 'plaza2': (222, 226, 232), 'deck': (176, 160, 160),
    })
    SEASON_PALETTES['winter'] = winter_pal

    if getattr(geo, 'SAT_OUTLINE', False):
        for is_fine, zone_id in ((False, SURFACE_IDS['zone2']), (True, SURFACE_IDS['zone'])):
            zone_polys = [Polygon(list(zip(*geo.convert_lonlat_to_local_meters(*np.array(z['ring']).T)))) for z in satellite_data['zones'] if bool(z.get('fine')) == is_fine]
            if zone_polys:
                for merged_zp in extract_polygons(shapely.unary_union(zone_polys)):
                    surface_draw.line(to_tex_coords(list(merged_zp.exterior.coords)), fill=zone_id, width=max(1, round(3.0 / tex_meters_per_px)))

    core_surface_grid = np.asarray(surface_img)[margin_px:margin_px + core_tex_width, margin_px:margin_px + core_tex_width]
    noise_primary = np.random.default_rng(1).random((core_tex_width, core_tex_width), dtype=np.float32)
    noise_secondary = np.random.default_rng(2).random((core_tex_width, core_tex_width), dtype=np.float32)

    NATURAL_SURFACE_IDS = [
        SURFACE_IDS[k] for k in (
            'grass', 'garden', 'park', 'meadow', 'forest', 'scrub', 'cemetery', 'wetland',
            'farmland', 'wheat', 'wheat2', 'rape', 'rape2', 'maize', 'maize2', 'soil', 'soil2',
        )
    ]
    is_natural_mask = np.isin(core_surface_grid, NATURAL_SURFACE_IDS)

    for season_name, palette in SEASON_PALETTES.items():
        color_lut = np.zeros((256, 3), np.uint8)
        for surf_name, surf_id in SURFACE_IDS.items():
            color_lut[surf_id] = palette[surf_name]
        tex_rgb = color_lut[core_surface_grid].astype(np.int16)

        dark_speckles = is_natural_mask & (noise_primary > 0.93)
        light_speckles = is_natural_mask & (noise_primary < 0.05)
        tex_rgb[dark_speckles] -= 14 if season_name != 'winter' else 10
        tex_rgb[light_speckles] += 14 if season_name != 'winter' else 6

        if season_name == 'spring':
            tex_rgb[is_natural_mask & (noise_secondary < 0.006)] = (240, 214, 70)
            tex_rgb[is_natural_mask & (noise_secondary > 0.995)] = (246, 246, 240)
        elif season_name == 'autumn':
            tex_rgb[is_natural_mask & (noise_secondary < 0.006)] = (196, 120, 46)
            tex_rgb[is_natural_mask & (noise_secondary > 0.997)] = (170, 74, 40)
        elif season_name == 'winter':
            road_mask = core_surface_grid == SURFACE_IDS['road']
            tex_rgb[road_mask & (noise_secondary < 0.06)] = (200, 204, 212)

        asphalt_mask = np.isin(core_surface_grid, [SURFACE_IDS['road'], SURFACE_IDS['parking']])
        tex_rgb[asphalt_mask & (noise_primary > 0.97)] += 8

        Image.fromarray(np.clip(tex_rgb, 0, 255).astype(np.uint8)).save(f'{out_dir}/ground_{season_name}.png')
        print('textura', season_name)

    # ---------------- Night Lightmap Generation ----------------
    print("Night Lightmap Generation")
    lightmap_res = 0.5
    lightmap_width = int(2 * HALF_SQUARE_SIDE / lightmap_res)
    lightmap_grid = np.zeros((lightmap_width, lightmap_width), np.float32)
    glow_radius = 14
    grid_y, grid_x = np.mgrid[-glow_radius * 2:glow_radius * 2 + 1, -glow_radius * 2:glow_radius * 2 + 1]
    glow_kernel = np.clip(1 - np.hypot(grid_x, grid_y) / (glow_radius * 2), 0, 1) ** 1.6

    for lx, ly, *_ in street_props['lamp']:
        px_cx, px_cy = int((lx + HALF_SQUARE_SIDE) / lightmap_res), int((HALF_SQUARE_SIDE - ly) / lightmap_res)
        x0, y0 = px_cx - glow_radius * 2, px_cy - glow_radius * 2
        x1, y1 = min(lightmap_width, x0 + glow_kernel.shape[1]), min(lightmap_width, y0 + glow_kernel.shape[0])
        sx0, sy0 = max(0, x0), max(0, y0)
        if x1 <= sx0 or y1 <= sy0:
            continue
        lightmap_grid[sy0:y1, sx0:x1] += glow_kernel[sy0 - y0:y1 - y0, sx0 - x0:x1 - x0]

    Image.fromarray((np.clip(lightmap_grid, 0, 1) * 255).astype(np.uint8)).save(f'{out_dir}/lightmap.png')

    # ---------------- Animal Herds ----------------
    print("Animal Herds")
    herds_output = []
    for herd_cfg in manual_overrides.get('stada', manual_overrides.get('herds', [])):
        raw_species = str(herd_cfg.get('typ', herd_cfg.get('type', 'cow'))).lower()
        species = 'horse' if any(k in raw_species for k in ('kun', 'kone', 'horse')) else 'deer' if any(k in raw_species for k in ('srn', 'deer')) else 'cow'
        hx, hy = geo.convert_lonlat_to_local_meters(float(herd_cfg['lon']), float(herd_cfg['lat']))
        radius = float(herd_cfg.get('radius', 30.0))
        herd_pt = Point(hx, hy)
        if extended_area_prep.contains(herd_pt) or herd_pt.buffer(radius).intersects(extended_area):
            hz = float(get_ground_elevation(hx, hy))
            if not is_tree_spot_free(hx, hy):
                candidates = [(hx + math.cos(a) * d, hy + math.sin(a) * d) for d in np.linspace(3, radius, 8) for a in np.linspace(0, math.tau, 12)]
                clear_spots = [pt for pt in candidates if is_tree_spot_free(*pt)]
                if clear_spots:
                    hx, hy = min(clear_spots, key=lambda pt: math.hypot(pt[0] - hx, pt[1] - hy))
                    hz = float(get_ground_elevation(hx, hy))
            herds_output.append({
                'typ': species, 'x': round(hx, 2), 'y': round(hy, 2), 'z': round(hz, 2),
                'n': int(herd_cfg.get('pocet', herd_cfg.get('count', 8))),
                'r': round(radius, 1),
                'mladata': bool(herd_cfg.get('mladata', True)),
            })

    # ---------------- Save scene.json ----------------
    def get_missing_neighbor_edges(chunk_coord):
        """Edges lacking a neighboring chunk get a soil cut skirt"""
        return ''.join(
            edge_name for edge_name, (delta_east, delta_north) in (('n', (0, 1)), ('s', (0, -1)), ('w', (-1, 0)), ('e', (1, 0)))
            if (chunk_coord[0] + delta_east, chunk_coord[1] + delta_north) not in MAP_CHUNKS)

    print("Save scene.json")
    max_scene_z = max([b['e'] + b['rh'] for b in scene_buildings] + [t[2] + t[3] for t in scene_trees] + [float(core_heights.max())])
    scene_payload = {
        'chunk': chunk,
        'center': [center_x, center_y],
        'half': HALF_SQUARE_SIDE,
        'step': grid_step,
        'n': grid_nodes,
        'hmin': base_elevation,
        'heights': [round(float(v), 2) for v in core_heights.ravel()],
        'ms': margin_steps,
        'hext': [round(float(v), 2) for v in ext_heights.ravel()],
        'nb': neighbor_offsets,
        'margin': CHUNK_MARGIN_METERS,
        'skirt': get_missing_neighbor_edges(chunk),
        'floor': TERRAIN_FLOOR_ELEVATION,
        'top': round(max_scene_z, 1),
        'buildings': scene_buildings,
        'trees': scene_trees,
        'props': street_props,
        'cars': parked_cars,
        'buses': parked_buses,
        'portals': tunnel_portals,
        'lines': linear_features,
        'herds': herds_output,
        'bridges': [
            {'p': run, 'w': deck[4], 'k': deck[5], 'c': deck[6], 'wtr': deck[8] if len(deck) > 8 else 0}
            for deck in bridge_decks
            if deck[0].intersects(extended_area)
            for run in clip_deck_runs([[*pt, round(float(gz), 2)] for pt, gz in zip(serialize_deck_line(deck), deck[3])])
        ],
    }
    json.dump(scene_payload, open(f'{out_dir}/scene.json', 'w'), separators=(',', ':'))

    # ---------------- Navigation Graph, Traffic & Info Metadata ----------------
    print("Navigation Graph, Traffic & Info Metadata")
    street_geoms_by_name = {}
    for line, _, _, name in non_rail_lines:
        if name:
            street_geoms_by_name.setdefault(name, []).append(line)

    street_labels_out = []
    for name, line_list in street_geoms_by_name.items():
        merged = shapely.MultiLineString(line_list).intersection(chunk_box)
        if merged.is_empty:
            continue
        mid_pt = (
            merged.interpolate(0.5, normalized=True)
            if merged.geom_type == 'LineString'
            else max(extract_linestrings(merged), key=lambda q: q.length).interpolate(0.5, normalized=True)
        )
        street_labels_out.append({'name': name, 'x': round(mid_pt.x, 1), 'y': round(mid_pt.y, 1), 'z': round(float(get_ground_elevation(mid_pt.x, mid_pt.y)), 1)})

    def is_empty_restriction(val):
        return val is None or (not isinstance(val, str) and hasattr(val, '__len__') and len(val) == 0)

    def get_oneway_direction(row):
        restrictions = row.get('access_restrictions')
        if restrictions is None or isinstance(restrictions, float):
            return 0
        for r in restrictions:
            when = r.get('when') or {}
            if (
                r.get('access_type') == 'denied'
                and r.get('between') is None
                and is_empty_restriction(when.get('mode'))
                and is_empty_restriction(when.get('using'))
                and is_empty_restriction(when.get('recognized'))
            ):
                if when.get('heading') == 'backward':
                    return 1
                if when.get('heading') == 'forward':
                    return -1
        return 0

    raw_oneways = [(sub_l, get_oneway_direction(row), row['class']) for _, row in segments_df.iterrows() if row['class'] in DRIVABLE_CLASSES and row['class'] != 'service' for sub_l in extract_linestrings(row.geom)]
    raw_oneways += [(sub_l, get_oneway_direction(row), row['class']) for geom, row in raw_bridge_records if row['class'] in DRIVABLE_CLASSES and row['class'] != 'service' for sub_l in extract_linestrings(geom)]

    traffic_cfg = geo.CITY_CONFIG.get('doprava')
    traffic_volumes, truck_shares = {}, {}
    if traffic_cfg:
        selected_sections = traffic_cfg.get('useky', {})
        for count_rec in json.load(open('build/scitani.json', encoding='utf-8')):
            is_chosen = count_rec['usek'] in selected_sections[count_rec['sil']] if count_rec['sil'] in selected_sections else count_rec['kraj'] == traffic_cfg['kraj']
            if is_chosen:
                traffic_volumes.setdefault(count_rec['sil'], []).append((count_rec['sv'], count_rec['nak']))
        truck_shares = {k: round(100 * sum(n for _, n in v) / max(1, sum(s for s, _ in v)), 1) for k, v in traffic_volumes.items()}
        traffic_volumes = {k: float(np.median([s for s, _ in v])) for k, v in traffic_volumes.items()}

    DEFAULT_TRAFFIC_BY_CLASS = {
        'trunk': 10000, 'primary': 8000, 'secondary': 3000, 'tertiary': 1500,
        'unclassified': 400, 'residential': 300, 'living_street': 100,
    }

    def get_segment_traffic(row):
        routes = row.get('routes')
        for route in (routes if routes is not None and not isinstance(routes, float) else []):
            ref = str(route.get('ref') or '').replace(' ', '')
            if ref in traffic_volumes:
                return traffic_volumes[ref], truck_shares[ref], True
        return DEFAULT_TRAFFIC_BY_CLASS.get(row['class'], 300), 0, False

    traffic_segments = []
    if traffic_cfg:
        for _, row in segments_df.iterrows():
            if row['class'] not in DRIVABLE_CLASSES or row['class'] == 'service':
                continue
            is_link = row.get('subclass') == 'link'
            for sub_l in extract_linestrings(row.geom):
                traffic_segments.append([sub_l, *get_segment_traffic(row), is_link, row['class']])
        for r_cls in {item[5] for item in traffic_segments}:
            measured = [item for item in traffic_segments if item[5] == r_cls and item[3] and not item[4]]
            if measured:
                med_vol = float(np.median([item[1] for item in measured]))
                med_truck = float(np.median([item[2] for item in measured]))
                for item in traffic_segments:
                    if item[5] == r_cls and not item[3] and not item[4]:
                        item[1], item[2] = med_vol, med_truck
        for item in traffic_segments:
            if item[4]:
                item[1] = item[1] * 0.2

    traffic_tree = shapely.STRtree([item[0] for item in traffic_segments]) if traffic_segments else None

    def lookup_piece_traffic(line, road_cls):
        mid = line.interpolate(0.5, normalized=True)
        nearby = list(traffic_tree.query(mid.buffer(3.0)))
        matching_cls = [k for k in nearby if traffic_segments[k][5] == road_cls] or nearby
        if not matching_cls:
            return None, 0, False
        best_item = traffic_segments[min(matching_cls, key=lambda k: traffic_segments[k][0].distance(mid))]
        return best_item[1], best_item[2], best_item[4]

    short_oneways = [line for line, _, _ in raw_oneways if line.length < 60]
    roundabout_rings = []
    for poly in shapely.ops.polygonize(shapely.ops.unary_union(short_oneways) if short_oneways else []):
        if poly.length < 250 and 4 * math.pi * poly.area / poly.length ** 2 > 0.85 and len(poly.exterior.coords) >= 8:
            roundabout_rings.append(shapely.geometry.polygon.orient(poly, 1.0).exterior)

    def get_relative_direction(ref_line, sub_line):
        pos_a = ref_line.project(sub_line.interpolate(0.25, normalized=True))
        pos_b = ref_line.project(sub_line.interpolate(0.75, normalized=True))
        return 1 if pos_b >= pos_a else -1

    resolved_oneways = []
    for line, direction, _ in raw_oneways:
        mid = line.interpolate(0.5, normalized=True)
        for ring in roundabout_rings:
            if line.length < 60 and ring.distance(mid) < 0.5 and ring.distance(Point(line.coords[0])) < 0.5 and ring.distance(Point(line.coords[-1])) < 0.5:
                direction = get_relative_direction(ring, line)
        if direction:
            resolved_oneways.append((line, direction))

    oneway_tree = shapely.STRtree([item[0] for item in resolved_oneways])

    def lookup_piece_oneway(line):
        mid = line.interpolate(0.5, normalized=True)
        for idx in oneway_tree.query(mid.buffer(0.5)):
            src_line, direction = resolved_oneways[idx]
            if src_line.distance(mid) < 0.5:
                return direction * get_relative_direction(src_line, line)
        return 0

    def extract_speed_limit(row):
        limits = row.get('speed_limits')
        if limits is None or isinstance(limits, float):
            return 0
        for lim in limits:
            max_spd = lim.get('max_speed')
            if max_spd and not lim.get('when') and lim.get('between') is None and max_spd.get('unit') == 'km/h':
                return int(max_spd['value'])
        return 0

    speed_limit_records = [(sub_l, extract_speed_limit(row)) for _, row in segments_df.iterrows() if row['class'] in DRIVABLE_CLASSES and extract_speed_limit(row) for sub_l in extract_linestrings(row.geom)]
    speed_limit_tree = shapely.STRtree([item[0] for item in speed_limit_records]) if speed_limit_records else None

    def lookup_piece_speed_limit(line):
        if speed_limit_tree is None:
            return 0
        mid = line.interpolate(0.5, normalized=True)
        for idx in speed_limit_tree.query(mid.buffer(0.5)):
            if speed_limit_records[idx][0].distance(mid) < 0.5:
                return speed_limit_records[idx][1]
        return 0

    def build_road_graph_edge(coords_3d, line, road_cls, road_name):
        edge = {'p': coords_3d, 'c': road_cls, 'n': road_name}
        oneway = lookup_piece_oneway(line)
        if oneway:
            edge['o'] = oneway
        speed_lim = lookup_piece_speed_limit(line)
        if speed_lim:
            edge['l'] = speed_lim
        if traffic_tree is not None and road_cls != 'service':
            vol, trucks, is_link = lookup_piece_traffic(line, road_cls)
            edge['v'] = int(vol or DEFAULT_TRAFFIC_BY_CLASS.get(road_cls, 300))
            if trucks:
                edge['t'] = trucks
            if is_link:
                edge['x'] = 1
        return edge

    road_graph = [
        build_road_graph_edge(
            [[round(x, 1), round(y, 1), round(float(z), 1)] for (x, y), z in zip(sub_l.coords, get_ground_elevation(*np.array(sub_l.coords).T))],
            sub_l,
            r_cls,
            name,
        )
        for orig_l, _, r_cls, name in drivable_lines
        for sub_l in extract_linestrings(orig_l.intersection(chunk_box.intersection(map_clip_area.buffer(-4))))
        if sub_l.length > 2
    ]

    for deck in bridge_decks:
        if deck[6] in DRIVABLE_CLASSES and chunk_box.intersects(deck[0]):
            coords_3d = serialize_deck_line(deck)
            deck_2d = LineString([(x, y) for x, y, _ in coords_3d])
            road_graph.append({**build_road_graph_edge(coords_3d, deck_2d, deck[6], deck[7]), '_deck': 1})

    def split_3d_polyline(coords_3d, split_dist, junction_xy):
        part_a, part_b, accum_dist = [coords_3d[0]], None, 0.0
        for i in range(1, len(coords_3d)):
            p0, p1 = coords_3d[i - 1], coords_3d[i]
            seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if part_b is None and accum_dist + seg_len >= split_dist:
                ratio = (split_dist - accum_dist) / (seg_len or 1.0)
                mid_pt = [junction_xy[0], junction_xy[1], round(p0[2] + (p1[2] - p0[2]) * ratio, 1)]
                part_a.append(mid_pt)
                part_b = [mid_pt, p1]
            elif part_b is None:
                part_a.append(p1)
            else:
                part_b.append(p1)
            accum_dist += seg_len
        return part_a, part_b

    # Snap T-junction endpoints onto mid-segments in the road graph
    print("Snap T-junction endpoints onto mid-segments in the road graph")
    for _ in range(3):
        edge_lines_2d = [LineString([(pt[0], pt[1]) for pt in edge['p']]) for edge in road_graph]
        edge_tree = shapely.STRtree(edge_lines_2d)
        pending_splits = {}
        for idx, edge in enumerate(road_graph):
            for end_pt in (edge['p'][0], edge['p'][-1]):
                pt_geom = Point(end_pt[0], end_pt[1])
                for target_idx in edge_tree.query(pt_geom.buffer(1.0)):
                    if target_idx == idx or edge_lines_2d[target_idx].distance(pt_geom) > 1.0:
                        continue
                    proj_dist = edge_lines_2d[target_idx].project(pt_geom)
                    if not 2.0 < proj_dist < edge_lines_2d[target_idx].length - 2.0:
                        continue
                    snapped_pt = split_3d_polyline(road_graph[target_idx]['p'], proj_dist, (end_pt[0], end_pt[1]))[0][-1]
                    if abs(snapped_pt[2] - end_pt[2]) < 2.0:
                        pending_splits.setdefault(int(target_idx), {})[round(proj_dist, 1)] = (end_pt[0], end_pt[1])
        if not pending_splits:
            break
        updated_graph = []
        for idx, edge in enumerate(road_graph):
            if idx not in pending_splits:
                updated_graph.append(edge)
                continue
            rem_coords, consumed_dist = edge['p'], 0.0
            for split_d in sorted(pending_splits[idx]):
                first_half, second_half = split_3d_polyline(rem_coords, split_d - consumed_dist, pending_splits[idx][split_d])
                if second_half is None:
                    break
                updated_graph.append({**edge, 'p': first_half})
                rem_coords, consumed_dist = second_half, split_d
            updated_graph.append({**edge, 'p': rem_coords})
        print('napojení doprostřed kusu rozděleno', sum(len(v) for v in pending_splits.values()))
        road_graph = updated_graph

    print('jednosměrných úseků pro auta', sum(1 for g in road_graph if g.get('o')), 'kruhových objezdů', len(roundabout_rings))

    if os.path.exists('assets/prejezdy.json'):
        sys.path.insert(0, 'nastroje')
        # TODO: does this exist?
        import prejezdy as rail_crossings_mod
        crossing_pts = [tuple(float(v) for v in geo.convert_lonlat_to_local_meters(lon, lat)) for lat, lon in json.load(open('assets/prejezdy.json'))]
        rail_lines_tree = shapely.STRtree([sub_l for _, geom, r_cls, _, _ in road_entries if r_cls == 'rail' for sub_l in extract_linestrings(geom)] or [Point(1e9, 1e9)])
        for edge in road_graph:
            if edge.get('c') == 'service' or edge.get('_deck'):
                continue
            crossings_on_edge = [q for q in rail_crossings_mod.na_kusu(edge['p'], crossing_pts) if len(rail_lines_tree.query(Point(q[1], q[2]), predicate='dwithin', distance=3.0))]
            if crossings_on_edge:
                edge['j'] = crossings_on_edge

    for edge in road_graph:
        edge.pop('_deck', None)
    print('železničních přejezdů', sum(len(g.get('j', [])) for g in road_graph))

    bus_stops = [
        {'name': get_primary_name(row), 'x': row.geom.x, 'y': row.geom.y}
        for _, row in infra_df.iterrows()
        if row['class'] == 'bus_stop' and chunk_box.contains(row.geom) and get_primary_name(row)
    ]

    def drape_line_to_ground(line):
        dense = line.segmentize(3.0)
        xs, ys = np.array(dense.coords).T
        return [[round(x, 1), round(y, 1), round(float(z), 1)] for x, y, z in zip(xs, ys, get_ground_elevation(xs, ys))]

    water_metadata = {'ponds': [], 'streams': []}
    for _, row in water_df.iterrows():
        clipped_water = row.geom.intersection(chunk_box)
        if clipped_water.is_empty:
            continue
        if clipped_water.geom_type in ('Polygon', 'MultiPolygon'):
            for poly in extract_polygons(clipped_water):
                if poly.area > 15:
                    water_metadata['ponds'].append(drape_line_to_ground(poly.exterior))
        elif row['class'] == 'river' or (row['class'] == 'stream' and render_streams):
            water_metadata['streams'] += [drape_line_to_ground(line) for line in extract_linestrings(clipped_water)]

    piste_paths = []
    for piste_line in ski_pistes:
        for clipped_piste in extract_linestrings(piste_line.intersection(chunk_box)):
            if clipped_piste.length < 30:
                continue
            draped = drape_line_to_ground(clipped_piste)
            piste_paths.append(draped if draped[0][2] >= draped[-1][2] else draped[::-1])

    tram_lines = [sub_l for _, row in segments_df.iterrows() if row.subtype == 'rail' and row['class'] == 'tram' for sub_l in extract_linestrings(row.geom)]
    tram_paths = (
        [drape_line_to_ground(clipped) for line in tram_lines for clipped in extract_linestrings(line.intersection(chunk_box)) if clipped.length > 1]
        + [serialize_deck_line(deck) for deck in bridge_decks if deck[6] == 'tram' and chunk_box.intersects(deck[0])]
    )
    tram_tree = shapely.STRtree(tram_lines) if tram_lines else None
    tram_stops = [
        [round(row.geom.x, 1), round(row.geom.y, 1), get_primary_name(row)]
        for _, row in infra_df.iterrows()
        if tram_tree is not None
        and row['class'] == 'stop_position'
        and row.geom.geom_type == 'Point'
        and get_primary_name(row)
        and chunk_box.contains(row.geom)
        and len(tram_tree.query(row.geom.buffer(4)))
    ]

    walk_paths = [
        drape_line_to_ground(clipped)
        for line, _, r_cls, _ in non_rail_lines
        if r_cls in ('footway', 'path', 'pedestrian', 'cycleway') and line.intersects(chunk_box)
        for clipped in [line.intersection(chunk_box)]
        if clipped.geom_type == 'LineString' and clipped.length > 8
    ]

    info_payload = {
        'buildings': building_metadata,
        'chimneys': residential_chimneys,
        'streets': street_labels_out,
        'roads': road_graph,
        'stops': bus_stops,
        'water': water_metadata,
        'walks': walk_paths,
        'pistes': piste_paths,
        'trams': tram_paths,
        'tstops': tram_stops,
        'decks': [{'p': serialize_deck_line(deck), 'w': deck[4]} for deck in bridge_decks if deck[5] != 'foot' and chunk_box.intersects(deck[0])],
        'lamps': street_props['lamp'],
        'ground_small': [
            [round(float(get_ground_elevation(x, y)), 1) for x in np.linspace(-HALF_SQUARE_SIDE, HALF_SQUARE_SIDE, 81)]
            for y in np.linspace(HALF_SQUARE_SIDE, -HALF_SQUARE_SIDE, 81)
        ],
    }
    json.dump(info_payload, open(f'{out_dir}/info_src.json', 'w'), separators=(',', ':'), ensure_ascii=False)
    print('hotovo')


if __name__ == '__main__':
    cli_data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    cli_chunk = sys.argv[2] if len(sys.argv) > 2 else '0,0'
    prepare_scene(cli_data_dir, cli_chunk)