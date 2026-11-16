import json
import os

# City config, and cached values
CITY_FILE_NAME = 'mesto.json'
CITY_CONFIG = json.load(open(CITY_FILE_NAME, encoding='utf-8'))

CHUNK_MARGIN_METERS = 80.5
CROP_TYPES = ['wheat'] * 4 + ['rape'] * 2 + ['maize'] * 2 + ['soil'] * 2
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
DEFAULT_TRAFFIC_BY_CLASS = {
    'trunk': 10000, 'primary': 8000, 'secondary': 3000, 'tertiary': 1500,
    'unclassified': 400, 'residential': 300, 'living_street': 100,
}
DRIVABLE_CLASSES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'residential', 'unclassified', 'living_street', 'service',
}
FLAT_CLASSES = {
    'garage', 'garages', 'apartments', 'industrial', 'warehouse', 'commercial', 'retail', 'school',
    'university', 'hospital', 'office', 'carport', 'parking', 'service', 'transportation',
    'sports_hall', 'sports_centre', 'greenhouse', 'roof',
}
FLAT_ROOF_PALETTE = [(141, 141, 141), (122, 122, 122), (156, 154, 146), (110, 112, 116)]
HALF_SQUARE_SIDE = float(CITY_CONFIG.get('polovina', 1150.0))   # Half-width of the square in meters
LAND_COVER_MAP = {
    'forest': 'forest', 'wood': 'forest', 'scrub': 'scrub', 'shrub': 'scrub',
    'grassland': 'meadow', 'wetland': 'wetland', 'bare_rock': 'rock', 'rock': 'rock',
}
LAND_USE_MAP = {
    'residential': 'garden', 'garden': 'garden', 'grass': 'park', 'park': 'park', 'village_green': 'park',
    'recreation_ground': 'park', 'meadow': 'meadow', 'farmland': 'farmland', 'allotments': 'garden',
    'cemetery': 'cemetery', 'pitch': 'pitch', 'track': 'track', 'playground': 'sand', 'stadium': 'pitch',
    'school': 'paved', 'kindergarten': 'garden', 'industrial': 'industrial', 'garages': 'industrial',
    'commercial': 'paved', 'retail': 'paved', 'pedestrian': 'cobble', 'plaza': 'cobble',
    'construction': 'construction', 'railway': 'dirt', 'barracks': 'industrial',
    'downhill': 'piste', 'ski_jump': 'skijump',
}
LAND_USE_TREE_SPACING = {
    'park': 11, 'cemetery': 9, 'residential': 16, 'garden': 12, 'grass': 20,
    'village_green': 18, 'recreation_ground': 20, 'allotments': 14, 'kindergarten': 14,
}
LINEAR_CLASS_MAP = {
    'fence': 'fence', 'hedge': 'hedge', 'wall': 'wall', 'retaining_wall': 'wall',
    'power_line': 'power', 'minor_line': 'power',
    'chair_lift': 'lift', 'drag_lift': 'lift', 't-bar': 'lift', 'platter': 'lift', 'cable_car': 'lift', 'gondola': 'lift',
}
MAP_CHUNKS = {tuple(chunk_coord): 'hotovo' for chunk_coord in CITY_CONFIG['ctverce']}
MAX_BRIDGE_GRADE = {'road': 0.06, 'rail': 0.03, 'foot': 0.08}
# TODO: only used once, maybe not necessary?
MIN_BRIDGE_LEN = 8.0
# TODO: only used once, maybe not necessary?
MIN_TUNNEL_LEN = 60.0
NO_TREE_LAND_USES = {
    'pitch', 'track', 'playground', 'stadium', 'school', 'industrial', 'garages', 'commercial',
    'retail', 'pedestrian', 'plaza', 'construction', 'railway', 'downhill', 'ski_jump',
}
PANEL_WALL_PALETTE = [
    (232, 226, 208), (213, 217, 222), (236, 236, 230), (222, 224, 222),
    (226, 199, 154), (169, 195, 201), (238, 232, 218), (204, 210, 214),
]
PEDESTRIAN_CLASSES = {'footway', 'path', 'cycleway', 'steps', 'pedestrian', 'bridleway', 'track'}
PITCHED_CLASSES = {
    'church', 'house', 'detached', 'semidetached_house', 'residential', 'terrace', 'farm', 'chapel',
    'bungalow', 'villa', 'cabin', 'religious', 'civic', 'hut', 'barn', 'farm_auxiliary',
    'kindergarten', 'allotment_house',
}
PITCHED_ROOF_PALETTE = [
    (168, 68, 47), (150, 60, 44), (142, 58, 42), (107, 59, 47),
    (74, 74, 82), (63, 74, 90), (156, 90, 60), (128, 52, 40),
]
REFERENCE_ELEVATION = float(CITY_CONFIG['href'])          # Zero terrain elevation reference (meters above sea level)
ROAD_HIERARCHY = {
    'service': 0, 'living_street': 1, 'residential': 2, 'unclassified': 2,
    'tertiary': 3, 'secondary': 4, 'primary': 5, 'trunk': 6, 'motorway': 7,
}
ROAD_SURFACE_MAP = {
    'track': 'dirt', 'path': 'dirt', 'bridleway': 'dirt', 'footway': 'footway',
    'cycleway': 'footway', 'steps': 'paved', 'pedestrian': 'cobble',
}
ROAD_WIDTHS = {
    'motorway': 18, 'trunk': 12, 'primary': 10, 'secondary': 9, 'tertiary': 8,
    'residential': 6, 'unclassified': 6, 'living_street': 5, 'service': 3.6,
    'pedestrian': 4, 'track': 3, 'footway': 1.8, 'path': 1.6, 'cycleway': 2,
    'steps': 2, 'bridleway': 1.6, 'unknown': 3,
}
ROOF_SHAPE_MAP = {
    'gabled': 'gabled', 'hipped': 'hipped', 'half_hipped': 'hipped', 'pyramidal': 'pyramidal',
    'saltbox': 'gabled', 'gambrel': 'gabled', 'skillion': 'gabled', 'mansard': 'hipped',
    'flat': 'flat', 'round': 'flat', 'dome': 'pyramidal',
}
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
    'spring': {
        'grass': (122, 176, 80), 'garden': (132, 184, 86), 'park': (118, 180, 78), 'meadow': (150, 188, 90),
        'forest': (78, 128, 62), 'scrub': (92, 128, 62), 'pitch': (80, 156, 72), 'track': (178, 92, 72),
        'sand': (214, 192, 138), 'paved': (196, 186, 160), 'industrial': (156, 152, 142), 'parking': (120, 120, 124),
        'road': (92, 92, 98), 'marking': (236, 232, 214), 'curb': (176, 172, 162), 'dirt': (170, 146, 102),
        'water': (64, 124, 190), 'pool': (92, 196, 230), 'stream': (72, 132, 196), 'rail': (88, 84, 82),
        'cemetery': (88, 132, 70), 'wetland': (100, 146, 92), 'rock': (150, 146, 136), 'construction': (164, 132, 96),
        'footway': (214, 206, 186), 'pool_edge': (236, 240, 240), 'farmland': (132, 150, 90), 'cobble': (200, 188, 164),
        'grave': (156, 152, 144), 'manhole': (66, 66, 70), 'court_blue': (70, 112, 170), 'zone': (255, 0, 220),
        'zone2': (0, 230, 230), 'piste': (150, 178, 92), 'skijump': (98, 160, 150),
        'wheat': (104, 146, 72), 'wheat2': (88, 128, 62), 'rape': (238, 214, 60), 'rape2': (216, 194, 50),
        'maize': (138, 106, 74), 'maize2': (112, 86, 60), 'soil': (130, 98, 68), 'soil2': (106, 80, 56),
        'plaza': (186, 186, 184), 'plaza2': (206, 206, 200), 'deck': (146, 104, 88),
    },
    'autumn': {
        'grass': (122, 154, 72), 'garden': (128, 158, 76), 'park': (120, 156, 70), 'meadow': (156, 160, 88),
        'forest': (92, 105, 58), 'scrub': (124, 116, 64), 'pitch': (80, 156, 72), 'track': (178, 92, 72),
        'sand': (214, 192, 138), 'paved': (196, 186, 160), 'industrial': (156, 152, 142), 'parking': (120, 120, 124),
        'road': (92, 92, 98), 'marking': (236, 232, 214), 'curb': (176, 172, 162), 'dirt': (170, 146, 102),
        'water': (64, 124, 190), 'pool': (92, 196, 230), 'stream': (72, 132, 196), 'rail': (88, 84, 82),
        'cemetery': (104, 128, 70), 'wetland': (100, 146, 92), 'rock': (150, 146, 136), 'construction': (164, 132, 96),
        'footway': (214, 206, 186), 'pool_edge': (236, 240, 240), 'farmland': (176, 150, 96), 'cobble': (200, 188, 164),
        'grave': (156, 152, 144), 'manhole': (66, 66, 70), 'court_blue': (70, 112, 170), 'zone': (255, 0, 220),
        'zone2': (0, 230, 230), 'piste': (150, 178, 92), 'skijump': (98, 160, 150),
        'wheat': (204, 182, 126), 'wheat2': (164, 132, 88), 'rape': (112, 150, 74), 'rape2': (98, 134, 64),
        'maize': (150, 126, 86), 'maize2': (124, 98, 66), 'soil': (122, 92, 62), 'soil2': (98, 72, 50),
        'plaza': (186, 186, 184), 'plaza2': (206, 206, 200), 'deck': (146, 104, 88),
    },
    'winter': {
        'grass': (234, 239, 246), 'garden': (234, 239, 246), 'park': (234, 239, 246), 'meadow': (234, 239, 246),
        'forest': (214, 222, 232), 'scrub': (220, 226, 234), 'pitch': (234, 239, 246), 'track': (234, 239, 246),
        'sand': (234, 239, 246), 'paved': (222, 226, 232), 'industrial': (206, 210, 216), 'parking': (176, 180, 188),
        'road': (104, 106, 114), 'marking': (190, 192, 196), 'curb': (226, 230, 236), 'dirt': (214, 214, 218),
        'water': (186, 210, 228), 'pool': (150, 168, 184), 'stream': (58, 96, 136), 'rail': (88, 84, 82),
        'cemetery': (234, 239, 246), 'wetland': (234, 239, 246), 'rock': (234, 239, 246), 'construction': (234, 239, 246),
        'footway': (226, 230, 236), 'pool_edge': (230, 234, 238), 'farmland': (234, 239, 246), 'cobble': (206, 210, 218),
        'grave': (222, 226, 232), 'manhole': (90, 92, 98), 'court_blue': (70, 112, 170), 'zone': (255, 0, 220),
        'zone2': (0, 230, 230), 'piste': (234, 239, 246), 'skijump': (234, 239, 246),
        'wheat': (234, 239, 246), 'wheat2': (212, 218, 228), 'rape': (234, 239, 246), 'rape2': (212, 218, 228),
        'maize': (234, 239, 246), 'maize2': (212, 218, 228), 'soil': (234, 239, 246), 'soil2': (212, 218, 228),
        'plaza': (208, 212, 220), 'plaza2': (222, 226, 232), 'deck': (176, 160, 160),
    },
}
SIDEWALK_ROADS = {'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified'}
STANDARD_WALL_PALETTE = [
    (232, 220, 192), (241, 234, 216), (217, 201, 163), (230, 211, 179),
    (214, 208, 196), (239, 230, 207), (216, 185, 143), (226, 196, 160), (205, 214, 206),
]
SURFACE_IDS = dict(
    grass=0, garden=1, park=2, meadow=3, forest=4, scrub=5, pitch=6, track=7, sand=8, paved=9,
    industrial=10, parking=11, road=12, marking=13, curb=14, dirt=15, water=16, pool=17, stream=18,
    rail=19, cemetery=20, wetland=21, rock=22, construction=23, footway=24, pool_edge=25, farmland=26,
    cobble=27, grave=28, manhole=29, court_blue=30, zone=31, zone2=32, piste=33, skijump=34,
    wheat=35, wheat2=36, rape=37, rape2=38, maize=39, maize2=40, soil=41, soil2=42,
    plaza=43, plaza2=44, deck=45,
)
TENEMENT_ROOF_PALETTE = [(70, 72, 80), (96, 64, 52), (150, 62, 44), (60, 64, 72), (128, 60, 46), (84, 88, 96)]
TENEMENT_WALL_PALETTE = [
    (232, 206, 150), (226, 186, 160), (206, 214, 180), (236, 226, 200), (214, 196, 170),
    (204, 202, 196), (238, 214, 170), (220, 176, 146), (196, 208, 214), (230, 220, 210),
]
TERRAIN_FLOOR_ELEVATION = float(CITY_CONFIG['dno'])   # Bottom of the soil cut for all chunks (below lowest map point)
