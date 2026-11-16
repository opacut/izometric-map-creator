"""Změří budovy z mračna bodů DMP 1G (body tříd 2 terén, 6 budova).

Může být spuštěno jako skript nebo importováno a voláno jako funkce:
    python dmp_vysky.py          # zpracuje LAZ soubory a vygeneruje json/csv
    python dmp_vysky.py --znovu  # znovu vyhodnotí již existující build/dmp_vysky.json
"""
import csv
import glob
import json
import math
import os
import sys
from multiprocessing import Pool
import laspy
import numpy as np
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

# Zajistí UTF-8 kódování a unbuffered stdout/stderr pro reálné logování
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

T = Transformer.from_crs(4326, 5514, always_xy=True)


def krovak(g):
    return shapely.transform(g, lambda c: np.column_stack(T.transform(c[:, 0], c[:, 1])))


def sheet_worker(args):
    """Worker funkce pro multiprocessing (vyžaduje globální proměnné P a RING)."""
    f, p_polys, ring_polys = args
    l = laspy.read(f)
    X, Y, Z, C = np.array(l.x), np.array(l.y), np.array(l.z), np.array(l.classification)
    out = {}

    for cls, polys, key in ((6, p_polys, 0), (2, ring_polys, 1)):
        m = C == cls
        if not m.any():
            continue
        tree = shapely.STRtree(shapely.points(X[m], Y[m]))
        zz = Z[m]
        pi, ti = tree.query(polys, predicate='contains')
        order = np.argsort(pi)
        pi, ti = pi[order]
        ti = ti[order]

        u, st = np.unique(pi, return_index=True)
        st = list(st) + [len(pi)]
        for j, k in enumerate(u):
            out.setdefault(int(k), [[], []])[key] = zz[ti[st[j]:st[j + 1]]].tolist()

    return out


def process_dmp_heights(reevaluate=False):
    """Hlavní funkce pro změření výšek budov z DMP 1G."""
    B = {}
    for f in sorted(glob.glob('data*/building.parquet')):
        for r in pq.read_table(f, columns=['id', 'geometry', 'height', 'num_floors', 'class']).to_pylist():
            if r['id'] in B:
                continue
            B[r['id']] = r

    ids = list(B)
    P = krovak(shapely.from_wkb([B[i]['geometry'] for i in ids]))
    P = shapely.make_valid(P)
    RING = shapely.difference(shapely.buffer(P, 14), shapely.buffer(P, 1))

    acc = {}
    fs = sorted(glob.glob('build/dmp/*.laz')) if not reevaluate else []

    if fs:
        tasks = [(f, P, RING) for f in fs]
        with Pool(8) as p:
            for i, o in enumerate(p.imap_unordered(sheet_worker, tasks)):
                for k, (a, b) in o.items():
                    q = acc.setdefault(k, [[], []])
                    q[0] += a
                    q[1] += b
                print(f"{i + 1}/{len(fs)} listů zpracováno")

        res = {}
        for k, (a, b) in acc.items():
            r = B[ids[k]]
            c = shapely.from_wkb(r['geometry']).centroid
            if len(a) >= 1 and len(b) >= 2:
                g0 = float(np.median(b))
                res[ids[k]] = [
                    round(float(P[k].area)),
                    len(a),
                    round(float(np.median(a)) - g0, 1),
                    round(float(np.percentile(a, 90)) - g0, 1),
                    r['num_floors'],
                    round(r['height'], 1) if r['height'] else None,
                    r['class'],
                    round(c.y, 6),
                    round(c.x, 6),
                ]
    else:
        if os.path.exists('build/dmp_vysky.json'):
            res = json.load(open('build/dmp_vysky.json', encoding='utf-8'))
        else:
            res = {}

    os.makedirs('build', exist_ok=True)
    json.dump(res, open('build/dmp_vysky.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"{len(B)} budov, {len(res)} změřeno")

    ok, bad = {}, []
    for k, (ar, n, med, p90, fl, hd, cl, la, lo) in res.items():
        why = ''
        if ar >= 80 and n < max(3, ar * 0.04):
            why = 'málo bodů'
        if med < 1.5 or med > 75:
            why = 'nesmyslná výška'
        if fl and fl >= 3 and abs(med - fl * 2.9) > 6.5 and (med < 3 or n < ar * 0.4):
            why = why or 'nesedí na patra OSM'
        if p90 - med > 8 and ar > 300:
            why = why or 'více výšek pod jedním půdorysem'

        if why:
            bad.append([why, cl or '', ar, n, med, p90, fl or '', la, lo, k])
        else:
            ok[k] = [med, p90]

    os.makedirs('assets', exist_ok=True)
    json.dump(ok, open('assets/dmp_vysky.json', 'w', encoding='utf-8'), separators=(',', ':'))

    with open('build/dmp_kontrola.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(['důvod', 'typ', 'plocha m2', 'bodů', 'medián', 'p90', 'patra OSM', 'lat', 'lon', 'id'])
        w.writerows(bad)

    print(f"použito {len(ok)}, ke kontrole {len(bad)} (build/dmp_kontrola.csv)")


if __name__ == '__main__':
    reeval = '--znovu' in sys.argv
    process_dmp_heights(reevaluate=reeval)