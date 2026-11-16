"""Postaví celé město z mesto.json: data -> výšky -> les a pole -> scény -> render -> dlaždice -> web (složka site/).

Použití:
    python postav.py          # všechno (kroky, které už mají výsledek, se přeskočí)
    python postav.py render   # jen scény + render + web (po úpravě pravidel nebo dominant)
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Přidání kořenové složky do sys.path, aby fungovaly lokální moduly
# TODO: I think we can do this differently
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent))

import geo
import render_par
import fetch_data
import dmp_fetch
import dmp_vysky

# Ensure UTF-8 output encoding and unbuffered stdout
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(line_buffering=True)

def main():
    os.chdir(script_dir)
    
    print("Starting...")
    
    args = sys.argv[1:]
    is_render_only = len(args) > 0 and args[0] == "render"
    
    if not is_render_only:
        print("Fetching data...")
        fetch_data.fetch_all()
        
        if not os.path.exists("assets/dmp_vysky.json"):
            print("Fetching DMP sheets...")
            dmp_fetch.fetch_dmp_sheets()
            print("Processing DMP heights...")
            dmp_vysky.process_dmp_heights()
            
        if not os.path.exists("build/les_maska.png"):
            print("Generating forest mask...")
            subprocess.run([sys.executable, "nastroje/les.py"], check=True)
            
        if not os.path.exists("assets/zabaged.json"):
            print("Generating ZABAGED & fields...")
            subprocess.run([sys.executable, "nastroje/zabaged.py"], check=True)
            subprocess.run([sys.executable, "nastroje/pole.py"], check=True)

    # Vyčištění dlaždic
    tiles_dir = Path("site/tiles")
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    tiles_json = Path("site/tiles.json")
    if tiles_json.exists():
        tiles_json.unlink()

    print("Running render_par (scény, render a dlaždice)...")
    render_par.main()

    print("Rendering seasonal tree drawings...")
    for s in ["summer", "autumn", "winter", "spring"]:
        cmd = ["node", "render.mjs", "mpp=0.2", "dir=build", f"season={s}", "alpha=1", "stromy=site/stromy"]
        print("Running node as a cmd. Consider not doing that.")
        #with open(os.devnull, "wb") as devnull:
        #    subprocess.run(cmd, stdout=devnull, stderr=devnull, check=True)
        subprocess.run(cmd, check=True)

    print("Running preview.py...")
    subprocess.run([sys.executable, "preview.py"], check=True)

    print("Running overlay.py...")
    subprocess.run([sys.executable, "overlay.py"], check=True)

    print("Running mhd.py...")
    try:
        subprocess.run([sys.executable, "mhd.py"], check=True)
    except subprocess.CalledProcessError:
        print("mhd.py skončil s chybou, pokračuji dál...")

    print("Running set_status.py...")
    subprocess.run([sys.executable, "set_status.py", "hotovo"], check=True)

    print("Running build_html.py...")
    subprocess.run([sys.executable, "build_html.py"], check=True)

    print("Hotovo. Náhled: npm run nahled  a otevři http://localhost:8000")
    print("Izomapa je zdarma. Jestli ti ušetřila práci, kup autorovi kafe: https://donate.stripe.com/3cI6oA0PE7u65Y9cim8g000 ☕")

if __name__ == "__main__":
    main()