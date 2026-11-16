import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from './node_modules/three/build/three.module.js';

const parseArgs = () => {
  const ROOT = path.dirname(fileURLToPath(import.meta.url));
  const args = Object.fromEntries(process.argv.slice(2).map(a => a.split('=')));
  return {
    ROOT,
    args,
    mpp: args.mpp ?? '0.4',
    tile: args.tile ?? '1024',
    outDir: path.join(ROOT, args.out ?? 'build/tiles')
  };
};

async function launchBrowser() {
  const browser = await chromium.launch({ 
    args: [
      '--enable-gpu', 
      '--ignore-gpu-blocklist',
      '--allow-file-access-from-files'
    ] 
  });
  const page = await browser.newPage();
  
  page.on('console', (m) => console.log('[page]', m.text()));
  page.on('pageerror', (e) => { console.log('[pageerror]', e.message); process.exit(1); });

  return { browser, page };
}

function buildRenderUrl(ROOT, mpp, tile, args) {
  const url = new URL(`file://${path.join(ROOT, 'render.html')}`);
  const params = {
    mpp,
    tile,
    season: args.season ?? 'autumn',
    mode: args.mode ?? 'day',
    dir: args.dir ?? 'build',
    alpha: args.alpha ?? '0',
    skirt: args.skirt,
    gx0: args.gx0,
    gy1: args.gy1
  };

  for (const [key, val] of Object.entries(params)) {
    if (val !== undefined) url.searchParams.set(key, val);
  }

  return url.toString();
}

async function exportTrees(page, ROOT, args) {
  const { url: dataUrl, ...meta } = await page.evaluate(() => window.TREE_EXPORT());
  const treeDir = path.join(ROOT, args.stromy);
  
  fs.mkdirSync(treeDir, { recursive: true });
  fs.writeFileSync(path.join(treeDir, `${args.season}.png`), Buffer.from(dataUrl.split(',')[1], 'base64'));
  fs.writeFileSync(path.join(treeDir, `${args.season}.json`), JSON.stringify(meta));
}

const shouldRender = (i, j, only, list) => {
  if (only) return only[0] === i && only[1] === j;
  if (list) return list.has(`${i}_${j}`);
  return true;
};

async function renderTiles(page, info, outDir, args) {
  const markPath = path.join(outDir, 'rozpracovano');
  fs.writeFileSync(markPath, '');

  const only = args.only ? args.only.split(',').map(Number) : null;
  const list = args.tiles ? new Set(args.tiles.split(';')) : null;

  try {
    for (let j = 0; j < info.rows; j++) {
      for (let i = 0; i < info.cols; i++) {
        if (!shouldRender(i, j, only, list)) continue;
        const t = Date.now();
        const url = await page.evaluate(([a, b]) => window.renderTile(a, b), [i, j]);
        fs.writeFileSync(path.join(outDir, `${i}_${j}.png`), Buffer.from(url.split(',')[1], 'base64'));
        console.log(`dlaždice ${i},${j}`, (Date.now() - t) / 1000, 's');
      }
    }
  } finally {
    fs.rmSync(markPath, { force: true });
  }
}

async function main() {
  const { ROOT, args, mpp, tile, outDir } = parseArgs();
  fs.mkdirSync(outDir, { recursive: true });

  const { browser, page } = await launchBrowser();

  const t0 = Date.now();
  await page.goto(buildRenderUrl(ROOT, mpp, tile, args));
  await page.waitForFunction(() => window.READY === true, null, { timeout: 240000, polling: 1000 });
  
  const info = await page.evaluate(() => window.INFO);
  console.log('scéna připravena za', (Date.now() - t0) / 1000, 's', info);
  fs.writeFileSync(path.join(outDir, 'info.json'), JSON.stringify(info));

  if (args.stromy) {
    await exportTrees(page, ROOT, args);
  } else {
    await renderTiles(page, info, outDir, args);
  }

  await browser.close();
}

await main();