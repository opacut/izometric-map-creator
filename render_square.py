import os
import sys
import time
from pathlib import Path
import prep
from render_chunk import render_chunk
import dzi

# Ensure UTF-8 output encoding across environments
os.environ["PYTHONUTF8"] = "1"

def run_square(coordinate: str = "0,0"):
    # Change working directory to the script's location
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    c_safe = coordinate.replace(",", "_")
    d_dir = "data" if coordinate == "0,0" else f"data_c{c_safe}"
    
    # Parse coordinate tuple for dzi.py if needed
    chunk_tuple = tuple(int(v) for v in coordinate.split(",")) if coordinate != "0,0" else None
    
    start_time = time.time()

    # Step 1: prep.py (called directly as a Python function)
    print("Here we go - starting prep.py")
    try:
        prep.prepare_scene(d_dir, coordinate)
    except Exception as e:
        print(f"CHYBA {coordinate} v prep.py: {e}")
        sys.exit(1)
        
    # Step 2: render_chunk (called directly as a Python function)
    print("OK, next is render_chunk")
    try:
        render_chunk(coordinate)
    except Exception as e:
        print(f"CHYBA {coordinate} v render_chunk: {e}")
        sys.exit(1)
        
    # Step 3: dzi.py (called directly as a Python function)
    print("OK, next is dzi.py")
    try:
        dzi.build_tiles(chunk=chunk_tuple)
    except Exception as e:
        print(f"CHYBA dzi {coordinate}: {e}")
        sys.exit(1)

    elapsed = int(time.time() - start_time)
    print(f"hotovo {coordinate} za {elapsed} s")

if __name__ == "__main__":
    coord = sys.argv[1] if len(sys.argv) > 1 else "0,0"
    run_square(coord)