"""Stáhne data pro výřez: Overture Maps (S3, anonymně) + terén Terrarium (AWS Open Data).

Může být spuštěno jako skript nebo importováno a voláno jako funkce.
"""
import math
import os
from pathlib import Path
import time
import urllib.request
import pyarrow.dataset
import pyarrow.fs
import pyarrow.parquet
import pyarrow.compute
import geo
from constants import HALF_SQUARE_SIDE
from constants import MAP_CHUNKS


RELEASE = '2026-10-21.0'
LAYERS = [
    ('buildings', 'building'), 
    ('transportation', 'segment'), 
    ('base', 'land_use'), 
    ('base', 'water'),
    ('base', 'land_cover'), 
    ('base', 'infrastructure'), 
    ('base', 'land'), 
    ('addresses', 'address')
]


def build_bbox_filter(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float
) -> pyarrow.compute.Expression:
    """Create a PyArrow filter expression for bounding-box intersection."""
    # Zde operátor & funguje, protože jde o Dataset Expressions
    return (
        (pyarrow.dataset.field("bbox", "xmin") < max_lon)
        & (pyarrow.dataset.field("bbox", "xmax") > min_lon)
        & (pyarrow.dataset.field("bbox", "ymin") < max_lat)
        & (pyarrow.dataset.field("bbox", "ymax") > min_lat)
    )

def fetch_all() -> None:
    """Downloads all layers for the whole map and cuts it into squares from MAP_CHUNKS."""
    proxy = os.environ.get('HTTPS_PROXY')
    s3 = pyarrow.fs.S3FileSystem(anonymous=True, region='us-west-2', **({'proxy_options': proxy} if proxy else {}))

    all_dir = Path("data_all")
    data_dir = Path("data")
    dem_dir = data_dir / "dem"
    all_dir.mkdir(parents=True, exist_ok=True)
    dem_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download main Overture layers (with 400m buffer)
    buffered_bounds = geo.get_map_bounding_box_lonlat(400)
    buffered_filter = build_bbox_filter(*buffered_bounds)
    for theme, data_type in LAYERS:
        out_path = all_dir / f'{data_type}.parquet'
        if out_path.exists():
            print(f"Skipping {theme}/{data_type} (already exists).")
            continue

        print(f"Fetching {theme}/{data_type}...")
        start_time = time.perf_counter()

        s3_uri = f"overturemaps-us-west-2/release/{RELEASE}/theme={theme}/type={data_type}/"
        dataset = pyarrow.dataset.dataset(s3_uri, filesystem=s3, format="parquet")
        table = dataset.to_table(filter=buffered_filter)
        pyarrow.parquet.write_table(table, out_path)

        elapsed = time.perf_counter() - start_time
        print(f"Finished {data_type}: {table.num_rows:,} rows in {elapsed:.1f}s.")

    # 2. Download Terrarium DEM elevation tiles for the full buffered extent
    X0, X1, Y0, Y1 = buffered_bounds
    zoom = 14
    n = 2**zoom
    tx0 = int((X0 + 180) / 360 * n)
    tx1 = int((X1 + 180) / 360 * n)
    ty0 = int((1 - math.asinh(math.tan(math.radians(Y1))) / math.pi) / 2 * n)
    ty1 = int((1 - math.asinh(math.tan(math.radians(Y0))) / math.pi) / 2 * n)

    for x in range(tx0, tx1 + 1):
        for y in range(ty0, ty1 + 1):
            tile = dem_dir / f"{zoom}_{x}_{y}.png"
            if not tile.exists():
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png"
                urllib.request.urlretrieve(url, tile)

    # 3. Slice downloaded layers per geo chunk
    for c in sorted(MAP_CHUNKS):
        print(f"geo chunk: {c}")
        #cx, cy = geo.get_chunk_center(c)
        cx = c[0] * 2 * HALF_SQUARE_SIDE
        cy = c[1] * 2 * HALF_SQUARE_SIDE
        r = HALF_SQUARE_SIDE + 300
        (x0, x1), (y0, y1) = geo.convert_local_meters_to_lonlat([cx - r, cx + r], [cy - r, cy + r])
        x0, x1, y0, y1 = float(x0), float(x1), float(y0), float(y1)

        # Každý čtverec má svůj vlastní adresář (např. data_c-1_0)
        #chunk_data_dir = Path(geo.get_chunk_directory(c))
        chunk_data_dir = Path('build' if tuple(c) == (0, 0) else f'build/c{c[0]}_{c[1]}')
        chunk_data_dir.mkdir(parents=True, exist_ok=True)
        dem_chunk_dir = chunk_data_dir / "dem"
        dem_chunk_dir.mkdir(parents=True, exist_ok=True)

        for _, typ in LAYERS:
            table = pyarrow.parquet.read_table(all_dir / f"{typ}.parquet")
            box = table.column("bbox").combine_chunks()
            mask = pyarrow.compute.and_(
                pyarrow.compute.and_(
                    pyarrow.compute.less(box.field("xmin"), x1),
                    pyarrow.compute.greater(box.field("xmax"), x0)
                ),
                pyarrow.compute.and_(
                    pyarrow.compute.less(box.field("ymin"), y1),
                    pyarrow.compute.greater(box.field("ymax"), y0)
                )
            )
            pyarrow.parquet.write_table(table.filter(mask), chunk_data_dir / f"{typ}.parquet")

        # Stažení DEM pro tento konkrétní čtverec
        for x in range(tx0, tx1 + 1):
            for y in range(ty0, ty1 + 1):
                tile = dem_chunk_dir / f"{zoom}_{x}_{y}.png"
                if not tile.exists():
                    url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png"
                    urllib.request.urlretrieve(url, tile)

        print("square", c, "->", chunk_data_dir, flush=True)

    # 4. Optional extra layers (places / division_area)
    bounds = geo.bbox.lonlat(0) if hasattr(geo, 'bbox') and hasattr(geo.bbox, 'lonlat') else buffered_bounds
    bbox_filter = build_bbox_filter(*bounds)
        
    if not os.path.exists('data/places.parquet'):
        s3_uri = f'overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/'
        output_path = Path('data/places.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = pyarrow.dataset.dataset(s3_uri, filesystem=s3, format='parquet')
        places_table = dataset.to_table(filter=bbox_filter)
        pyarrow.parquet.write_table(places_table, output_path)

    if not os.path.exists('data_all/division_area.parquet'):
        s3_uri = f'overturemaps-us-west-2/release/{RELEASE}/theme=divisions/type=division_area/'
        output_path = Path('data_all/division_area.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = pyarrow.dataset.dataset(s3_uri, filesystem=s3, format='parquet')
        div_table = dataset.to_table(filter=bbox_filter)
        pyarrow.parquet.write_table(div_table, output_path)