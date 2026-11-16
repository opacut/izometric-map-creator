import json
import os

# City config, and cached values
CITY_FILE_NAME = 'mesto.json'
CITY_CONFIG = json.load(open(CITY_FILE_NAME, encoding='utf-8'))
HALF_SQUARE_SIDE = float(CITY_CONFIG.get('polovina', 1150.0))   # Half-width of the square in meters
MAP_CHUNKS = {tuple(chunk_coord): 'hotovo' for chunk_coord in CITY_CONFIG['ctverce']}
CHUNK_MARGIN_METERS = 80.5
TERRAIN_FLOOR_ELEVATION = float(CITY_CONFIG['dno'])   # Bottom of the soil cut for all chunks (below lowest map point)
REFERENCE_ELEVATION = float(CITY_CONFIG['href'])          # Zero terrain elevation reference (meters above sea level)