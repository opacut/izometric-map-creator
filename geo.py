"""Common coordinates and camera (must correspond to render.html)."""
import math
import numpy as np

from constants import CITY_CONFIG
from constants import HALF_SQUARE_SIDE
from constants import MAP_CHUNKS
#from constants import CHUNK_MARGIN_METERS

# City configuration: loaded from JSON file (default 'mesto.json', or override via IZO_MESTO environment variable)
#CITY_CONFIG = json.load(open(os.environ.get('IZO_MESTO', 'mesto.json'), encoding='utf-8'))
#_CENTER_LONGITUDE, _CENTER_LATITUDE = CITY_CONFIG['stred']          # Center of coordinate square (0, 0)
# only in fetch_data.py
#HALF_SQUARE_SIDE = float(CITY_CONFIG.get('polovina', 1150.0))   # Half-width of the square in meters
# not used anywhere
#REFERENCE_ELEVATION = float(CITY_CONFIG['href'])          # Zero terrain elevation reference (meters above sea level)

# Map is a grid of square chunks (2 * HALF_SQUARE_SIDE); 'east_index' grows east, 'north_index' grows north. Chunk (0, 0) must be present.
# refd in fetch_data.py and render_par.py
#MAP_CHUNKS = {tuple(chunk_coord): 'hotovo' for chunk_coord in CITY_CONFIG['ctverce']}
# only referenced from the outside
# refd in prep.py and nastroje
#CHUNK_MARGIN_METERS = 80.5           # Margin of adjacent chunks rendered inside this one (for cross-seam buildings, trees, shadows); equals 280 px texture
# not used anywhere
#SHOW_SAT_OUTLINE = False     # True = magenta borders for zones refined from satellite imagery (quality control)
# only referenced from prep.py
#TERRAIN_FLOOR_ELEVATION = float(CITY_CONFIG['dno'])   # Bottom of the soil cut for all chunks (below lowest map point)

METERS_PER_DEGREE_LONGITUDE = math.cos(math.radians(CITY_CONFIG['stred'][1])) * 111320.0
METERS_PER_DEGREE_LATITUDE = 110574.0

_CAMERA_ELEVATION_ANGLE = math.asin(math.tan(math.atan(0.5)))   # 30° -> 2:1 dimetric projection
_CAMERA_AZIMUTH_ANGLE = math.radians(225)                     # View from southwest
_CAMERA_VECTOR_Z = np.array([math.sin(_CAMERA_AZIMUTH_ANGLE) * math.cos(_CAMERA_ELEVATION_ANGLE), math.sin(_CAMERA_ELEVATION_ANGLE), -math.cos(_CAMERA_AZIMUTH_ANGLE) * math.cos(_CAMERA_ELEVATION_ANGLE)])
CAMERA_VECTOR_X = np.cross([0, 1, 0], _CAMERA_VECTOR_Z)
CAMERA_VECTOR_X /= np.linalg.norm(CAMERA_VECTOR_X)
CAMERA_VECTOR_Y = np.cross(_CAMERA_VECTOR_Z, CAMERA_VECTOR_X)


### SECTION: only refd from the outside
#def get_chunk_center(chunk_coord):
#    return chunk_coord[0] * 2 * HALF_SQUARE_SIDE, chunk_coord[1] * 2 * HALF_SQUARE_SIDE

#def get_chunk_directory(chunk_coord):
#    return 'build' if tuple(chunk_coord) == (0, 0) else f'build/c{chunk_coord[0]}_{chunk_coord[1]}'

def get_map_bounding_box_lonlat(margin=300.0):
    """Bounding box of the entire map in geographic coordinates: lon0, lon1, lat0, lat1"""
    min_x, min_y, max_x, max_y = get_map_extent_local_meters(margin)
    (lon_min, lon_max), (lat_min, lat_max) = convert_local_meters_to_lonlat([min_x, max_x], [min_y, max_y])
    return float(lon_min), float(lon_max), float(lat_min), float(lat_max)

def find_chunk_containing_point(local_x, local_y):
    """Returns the chunk coordinate containing a point given in local chunk (0, 0) coordinates"""
    return (
        int(math.floor((local_x + HALF_SQUARE_SIDE) / (2 * HALF_SQUARE_SIDE))), 
        int(math.floor((local_y + HALF_SQUARE_SIDE) / (2 * HALF_SQUARE_SIDE)))
    )

# only refd from prep.py
# TODO: study this comprehension
#def get_neighboring_chunks(chunk_coord):
#    """Returns neighboring chunks (including diagonals) that exist in the map: [(delta_east, delta_north), ...]"""
#    return [
#        (delta_east, delta_north) 
#        for delta_north in (1, 0, -1) 
#        for delta_east in (-1, 0, 1) 
#        if (delta_east or delta_north) and (chunk_coord[0] + delta_east, chunk_coord[1] + delta_north) in MAP_CHUNKS
#    ]

# only refd from prep.py
#def get_missing_neighbor_edges(chunk_coord):
#    """Edges lacking a neighboring chunk get a soil cut skirt"""
#    return ''.join(
#        edge_name for edge_name, (delta_east, delta_north) in (('n', (0, 1)), ('s', (0, -1)), ('w', (-1, 0)), ('e', (1, 0)))
#        if (chunk_coord[0] + delta_east, chunk_coord[1] + delta_north) not in MAP_CHUNKS)

def convert_lonlat_to_local_meters(longitude, latitude):
    return (
        (np.asarray(longitude) - CITY_CONFIG['stred'][0]) * METERS_PER_DEGREE_LONGITUDE, 
        (np.asarray(latitude) - CITY_CONFIG['stred'][1]) * METERS_PER_DEGREE_LATITUDE
    )


### SECTION: refd both internally and externally
def get_map_extent_local_meters(margin=0.0):
    """Bounding rectangle of the entire map in local meters: min_x, min_y, max_x, max_y"""
    east_indices = [chunk_coord[0] for chunk_coord in MAP_CHUNKS]
    north_indices = [chunk_coord[1] for chunk_coord in MAP_CHUNKS]
    return (
        min(east_indices) * 2 * HALF_SQUARE_SIDE - HALF_SQUARE_SIDE - margin, 
        min(north_indices) * 2 * HALF_SQUARE_SIDE - HALF_SQUARE_SIDE - margin,
        max(east_indices) * 2 * HALF_SQUARE_SIDE + HALF_SQUARE_SIDE + margin, 
        max(north_indices) * 2 * HALF_SQUARE_SIDE + HALF_SQUARE_SIDE + margin)

def convert_local_meters_to_lonlat(local_x, local_y):
    return (
        np.asarray(local_x) / METERS_PER_DEGREE_LONGITUDE + CITY_CONFIG['stred'][0], 
        np.asarray(local_y) / METERS_PER_DEGREE_LATITUDE + CITY_CONFIG['stred'][1]
    )

# only refd from build_html.py
def project_to_camera_view(local_x, local_y, height):
    """Local coordinates (east, north, height) -> camera plane coordinates + depth (smaller = closer)"""
    position_vector = np.stack([np.asarray(local_x, float), np.asarray(height, float), -np.asarray(local_y, float)], -1)
    return (
        position_vector @ CAMERA_VECTOR_X,
        position_vector @ CAMERA_VECTOR_Y,
        -(position_vector @ _CAMERA_VECTOR_Z)
    )


class Image2D:
    """Converts local coordinates to output image pixels (using renderer's info.json)"""
    def __init__(self, renderer_info):
        self.min_x = renderer_info['x0']
        self.max_y = renderer_info['y1']
        self.meters_per_pixel = renderer_info['mpp']

    def project_to_pixels(self, local_x, local_y, height):
        view_x, view_y, depth = project_to_camera_view(local_x, local_y, height)
        return (
            (view_x - self.min_x) / self.meters_per_pixel,
            (self.max_y - view_y) / self.meters_per_pixel,
            depth
        )
