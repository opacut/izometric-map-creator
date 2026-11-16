import sys
import base64
from pathlib import Path
from urllib.parse import urlencode
from playwright.sync_api import sync_playwright
import traceback
import subprocess
import time
import json


def render_single_variant(d_dir: Path, season: str, mode: str, alpha: str, gx0: str, gy1: str, mpp: str, tile: str, skirt: str | None, stromy: str | None):
    """Renders tiles using Playwright for Python."""

    root = Path(__file__).parent.resolve()
    out_dir = d_dir / ("tiles_lights" if mode == "lights" else f"tiles_{season}")
    out_dir.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        [
            "python3", "-m", "http.server",
            "--bind", "127.0.0.1",
            "--directory", str(root),
            "8001",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        time.sleep(0.5)

        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--enable-gpu", "--ignore-gpu-blocklist"]
            )

            page = browser.new_page()

            #page.on("console", lambda m: print(f"[page] {m.text}"))
            page.on("pageerror", lambda e: print(f"[pageerror] {e}"))

            page.on("console", lambda m: print(f"[page] {m.text}"))
            page.on("requestfailed", lambda r: print(f"[REQUEST FAILED] {r.url} : {r.failure}"))
            page.on("response", lambda r: print(f"[HTTP {r.status}] {r.url}" if r.status >= 400 else ""))

            params = {
                "mpp": mpp,
                "tile": tile,
                "season": season,
                "mode": mode,
                "dir": str(
                    d_dir.relative_to(root)
                    if d_dir.is_relative_to(root)
                    else d_dir
                ),
                "alpha": alpha,
            }

            if skirt:
                params["skirt"] = skirt

            if gx0:
                params["gx0"] = gx0
                params["gy1"] = gy1

            url = f"http://127.0.0.1:8001/render.html?{urlencode(params)}"

            t0 = time.time()
            page.goto(url)

            page.wait_for_function(
                "window.READY === true",
                timeout=240000,
                polling=1000,
            )

            info = page.evaluate("window.INFO")
            print(f"scéna připravena za {time.time() - t0:.2f} s", info)

            (out_dir / "info.json").write_text(
                json.dumps(info), encoding="utf-8"
            )

            if stromy:
                t = page.evaluate("window.TREE_EXPORT()")
                tree_dir = root / stromy
                tree_dir.mkdir(parents=True, exist_ok=True)

                data_url = t.pop("url")
                header, encoded = data_url.split(",", 1)

                (tree_dir / f"{season}.png").write_bytes(
                    base64.b64decode(encoded)
                )
                (tree_dir / f"{season}.json").write_text(
                    json.dumps(t), encoding="utf-8"
                )
                return

            mark_path = out_dir / "rozpracovano"
            mark_path.write_text("")

            try:
                for j in range(info["rows"]):
                    for i in range(info["cols"]):
                        t_tile = time.time()

                        tile_data_url = page.evaluate(
                            f"window.renderTile({i}, {j})"
                        )

                        _, encoded_tile = tile_data_url.split(",", 1)

                        (out_dir / f"{i}_{j}.png").write_bytes(
                            base64.b64decode(encoded_tile)
                        )

                        #print(
                        #    f"dlaždice {i},{j} "
                        #    f"{time.time() - t_tile:.2f} s"
                        #)
            finally:
                if mark_path.exists():
                    mark_path.unlink()

    finally:
        # Stop the HTTP server. Playwright manages browser cleanup
        # automatically when its context exits.
        server.terminate()
        server.wait()



def render_chunk(coordinate: str, variants: str = "autumn winter spring summer lights"):
    print("Entering render_chunk")
    
    if coordinate == "0,0":
        d_dir = Path("build")
    else:
        d_dir = Path(f"build/c{coordinate.replace(',', '_')}")
    
    d_dir.mkdir(parents=True, exist_ok=True)
    
    variant_list = variants.split()
    gx0 = "-1646.34559672906"
    gy1 = "988.7958500237901"
    
    
    for s in variant_list:
        
        season = "autumn" if s == "lights" else s
        mode = "lights" if s == "lights" else "day"
        
        try:
            render_single_variant(
                d_dir=d_dir,
                season=season,
                mode=mode,
                alpha="1",
                gx0=gx0,
                gy1=gy1,
                mpp="0.2",
                tile="1024",
                skirt=None,
                stromy=None
            )
        except Exception as e:
            print(f"RENDER SELHAL {coordinate} {s}: {e}")
            # TODO: implement better logging
            traceback.print_exc()
            sys.exit(1)
            
        print("done")
        print(1) # Tile count / success marker placeholder matching shell logic

    print(f"HOTOVO {coordinate}")

if __name__ == "__main__":
    coord = sys.argv[1] if len(sys.argv) > 1 else "0,0"
    vars_arg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "autumn winter spring summer lights"
    render_chunk(coord, vars_arg)