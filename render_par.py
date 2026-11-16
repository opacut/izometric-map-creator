"""Souběžný render: několik čtverců najednou, každý projde prep -> render -> dzi.

Použití:
    python render_par.py              # automatický počet vláken a všechny čtverce z MAP_CHUNKS
    python render_par.py 2            # ručně 2 čtverce najednou
    python render_par.py 2 "0,0 1,0"  # ručně 2 čtverce a pouze vybrané
"""
import os
import sys
import time
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import geo
from render_square import run_square
from constants import MAP_CHUNKS
import ctypes

# Ensure UTF-8 output encoding and unbuffered stdout for real-time logging
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(line_buffering=True)

def get_default_par():
    try:
        if sys.platform == 'darwin':
            mem = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']))
        elif sys.platform.startswith('linux'):
            mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        else:
            class M(ctypes.Structure):
                _fields_ = [('l', ctypes.c_ulong), ('m', ctypes.c_ulong), ('t', ctypes.c_ulonglong), ('a', ctypes.c_ulonglong)] + [(f'x{i}', ctypes.c_ulonglong) for i in range(5)]
            s = M(); s.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)); mem = s.t
    except Exception:
        mem = 8 << 30
    # na čtverec ~4 GB paměti, na čtverec aspoň 3 jádra; nejvýš 4
    return max(1, min(4, int(mem / (4 << 30)), (os.cpu_count() or 2) // 3))

def worker_run(coordinate: str):
    """Obalová funkce pro spuštění v samostatném procesu."""
    try:
        run_square(coordinate)
        return coordinate, True
    except Exception as e:
        print(f"CHYBA {coordinate}: {e}", file=sys.stderr)
        return coordinate, False

def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    os.makedirs("build", exist_ok=True)
    log_path = Path("build/render_par.log")

    # Zpracování argumentů příkazové řádky
    args = sys.argv[1:]
    par = None
    list_str = None

    if args:
        if args[0].isdigit():
            par = int(args[0])
            if len(args) > 1:
                list_str = args[1]
        else:
            list_str = args[0]

    if par is None:
        #par = get_default_par()
        par = 1

    if list_str:
        chunks = list_str.split()
    else:
        chunks = [f"{i},{j}" for i, j in sorted(MAP_CHUNKS.keys())]

    print(f"souběžně {par} čtverců")
    t0 = time.time()

    success = True
    #with ProcessPoolExecutor(max_workers=par) as executor:
    #    futures = {executor.submit(worker_run, coord): coord for coord in chunks}
    #    for future in as_completed(futures):
    #        coord, ok = future.result()
    #        if not ok:
    #            success = False
    # --- TEMPORARY FIX FOR DEBUGGING ---
    # Bypassing ProcessPoolExecutor so pdb can read from terminal
    for coord in chunks:
        print(f"Running synchronously for {coord}...")
        _, ok = worker_run(coord)
        if not ok:
            success = False

    elapsed_min = int((time.time() - t0) / 60)
    msg = f"render hotov za {elapsed_min} min"
    print(msg)

    # Zápis do logu a současný výstup na stdout
    log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()