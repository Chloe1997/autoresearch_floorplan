"""Plot how the auto-research run evolves: score per eval + running best, as a self-contained HTML page.

Reads work/history.jsonl (written by harness_baseline.py, one line per run_dev_eval).

usage
  python tools/plot_progress.py                    # latest run -> work/progress.html
  python tools/plot_progress.py --all              # every run in history, one after another
  python tools/plot_progress.py --watch 10         # rewrite every 10 s (page auto-reloads) while a run is going
  python tools/plot_progress.py --backfill         # score work/v*.py files that have no history entry yet
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
HISTORY = WORK / "history.jsonl"
BASELINE = 1.0


def load(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                   # half-written last line during a live run
    return rows


def backfill(tl, jobs):
    """Score versions in work/ that were produced before history logging existed."""
    sys.path.insert(0, str(ROOT / "fp"))
    from runner import score
    done = {r["version"] for r in load(HISTORY)}
    todo = sorted(p for p in glob.glob(str(WORK / "v*.py")) if Path(p).stem not in done)
    best, best_v = 0.0, "baseline"
    for i, p in enumerate(todo, 1):
        r = score(p, "dev", tl, ROOT / "instances", jobs)
        v = Path(p).stem
        if r["score"] > best:
            best, best_v = r["score"], v
        rec = {"run": "backfill", "t": os.path.getmtime(p), "eval": i, "version": v,
               "score": round(r["score"], 4), "best": round(best, 4), "best_version": best_v,
               "per_instance": {x["inst"]: round(x["score"], 4) for x in r["per_instance"]}}
        with open(HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i}/{len(todo)}] {v} score={rec['score']:.4f} best={best:.4f} ({best_v})")


def select(rows, all_runs):
    if not rows:
        return []
    if not all_runs:
        last = rows[-1]["run"]
        rows = [r for r in rows if r["run"] == last]
    out, best, best_v = [], float("-inf"), None
    for i, r in enumerate(rows):                       # recompute running best over what's shown
        if r["score"] > best:
            best, best_v = r["score"], r["version"]
        out.append(dict(r, idx=i, run_best=best, run_best_v=best_v))
    return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
__REFRESH__
<title>Harness Progress</title>
<style>
:root{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#8a8984;--grid:#e8e7e3;
  --series-1:#2a78d6;--raw:#a9a8a2;--ref:#c0392b;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#222221;--ink:#f0efec;
  --ink-2:#c3c2b7;--ink-3:#8f8e86;--grid:#333331;--series-1:#3987e5;--raw:#6d6c66;--ref:#e5675a;}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#222221;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#8f8e86;
  --grid:#333331;--series-1:#3987e5;--raw:#6d6c66;--ref:#e5675a;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--ink-2);margin:0 0 20px}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}
.tile{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:10px 14px;min-width:140px}
.tile .k{color:var(--ink-2);font-size:12px}.tile .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.card{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:12px 12px 4px;position:relative}
.legend{display:flex;flex-wrap:wrap;gap:16px;color:var(--ink-2);font-size:12px;margin:0 0 6px 4px}
.legend span{display:inline-flex;align-items:center;gap:6px}.legend svg{width:22px;flex:none}
svg{display:block;width:100%;height:auto;overflow:visible}
.axis text{fill:var(--ink-2);font-size:11px}.axis line,.axis path{stroke:var(--grid)}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--grid);border-radius:6px;
  padding:6px 9px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;white-space:nowrap}
.tip b{font-variant-numeric:tabular-nums}.tip .m{color:var(--ink-2)}
details{margin-top:20px}summary{cursor:pointer;color:var(--ink-2)}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;margin-top:8px}
th,td{padding:4px 10px;text-align:right;border-bottom:1px solid var(--grid)}th{color:var(--ink-2);font-weight:500}
td:first-child,th:first-child{text-align:left}
.empty{color:var(--ink-2);padding:40px 0;text-align:center}
</style></head><body><main>
<h1>Harness progress</h1>
<p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>
<div class="card">
  <div class="legend">
    <span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" stroke="var(--raw)" stroke-width="2"/><circle cx="11" cy="5" r="3" fill="var(--raw)"/></svg>Dev score per eval</span>
    <span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" stroke="var(--series-1)" stroke-width="2.5"/></svg>Running best</span>
    <span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" stroke="var(--ref)" stroke-width="1.5" stroke-dasharray="4 3"/></svg>Baseline 1.00</span>
  </div>
  <div id="chart"></div>
  <div class="tip" id="tip"></div>
</div>
<details><summary>Table view</summary><div class="tw" id="table"></div></details>
</main>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const fmt = (v, d=4) => v == null ? "–" : v.toFixed(d);
const ago = s => s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${(s/60).toFixed(1)} min` : `${(s/3600).toFixed(1)} h`;

if (!DATA.length) {
  $("chart").innerHTML = '<div class="empty">No evaluations yet — run harness_baseline.py (or --backfill).</div>';
} else {
  const t0 = DATA[0].t, last = DATA[DATA.length-1], bestRow = DATA.reduce((a,b) => b.score > a.score ? b : a);
  const runs = [...new Set(DATA.map(d => d.run))];
  $("sub").textContent = `${runs.length > 1 ? runs.length + " runs" : "run " + runs[0]} · ${DATA.length} evals · ` +
    `${ago(last.t - t0)} elapsed · generated ${new Date().toLocaleTimeString()}`;
  const fails = DATA.filter(d => d.score < 0.5).length;
  $("tiles").innerHTML = [
    ["Best dev score", fmt(bestRow.score)], ["Best version", bestRow.version],
    ["vs baseline", (bestRow.score >= 1 ? "+" : "") + ((bestRow.score - 1) * 100).toFixed(1) + "%"],
    ["Evals", DATA.length], ["Broken evals (<0.5)", fails]
  ].map(([k,v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  const W = 1060, H = 420, m = {l: 56, r: 92, t: 14, b: 44};
  const iw = W - m.l - m.r, ih = H - m.t - m.b, n = DATA.length;
  const ys = DATA.map(d => d.score).concat([1]);
  let lo = Math.min(...ys), hi = Math.max(...ys); const pad = (hi - lo) * 0.08 || 0.05; lo -= pad; hi += pad;
  const x = i => m.l + (n === 1 ? iw/2 : i * iw / (n - 1));
  const y = v => m.t + (hi - v) / (hi - lo) * ih;
  const ticks = (a, b, k) => { const s = Math.pow(10, Math.floor(Math.log10((b-a)/k))); const st = [1,2,5,10].map(f=>f*s).find(f => (b-a)/f <= k);
    const out=[]; for (let v = Math.ceil(a/st)*st; v <= b + 1e-9; v += st) out.push(+v.toFixed(6)); return out; };
  let g = "";
  for (const v of ticks(lo, hi, 6)) g += `<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/>` +
    `<text x="${m.l-8}" y="${y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`;
  const xstep = Math.max(1, Math.ceil(n / 12));
  for (let i = 0; i < n; i += xstep) g += `<text x="${x(i)}" y="${H-m.b+18}" text-anchor="middle">${i+1}</text>`;
  // run boundaries when several runs are shown
  let runsep = ""; for (let i = 1; i < n; i++) if (DATA[i].run !== DATA[i-1].run)
    runsep += `<line x1="${(x(i)+x(i-1))/2}" x2="${(x(i)+x(i-1))/2}" y1="${m.t}" y2="${H-m.b}" stroke="var(--ink-3)" stroke-dasharray="2 4"/>`;
  const raw = DATA.map((d,i) => `${i?"L":"M"}${x(i)},${y(d.score)}`).join("");
  const best = DATA.map((d,i) => `${i?"L":"M"}${x(i)},${y(d.run_best)}`).join("");
  const dots = DATA.map((d,i) => `<circle cx="${x(i)}" cy="${y(d.score)}" r="4" fill="var(--raw)" stroke="var(--panel)" stroke-width="2"/>`).join("") +
               DATA.map((d,i) => `<circle cx="${x(i)}" cy="${y(d.run_best)}" r="4" fill="var(--series-1)" stroke="var(--panel)" stroke-width="2"/>`).join("");
  const bi = bestRow.idx, bx = x(bi), by = y(bestRow.score);
  $("chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Dev score per eval and running best">
    <g class="axis">${g}<text x="${m.l + iw/2}" y="${H-6}" text-anchor="middle">Evaluation #</text>
      <text transform="translate(14 ${m.t + ih/2}) rotate(-90)" text-anchor="middle">Dev score (higher is better)</text></g>
    ${runsep}
    <line x1="${m.l}" x2="${W-m.r}" y1="${y(1)}" y2="${y(1)}" stroke="var(--ref)" stroke-width="1.5" stroke-dasharray="6 4"/>
    <path d="${raw}" fill="none" stroke="var(--raw)" stroke-width="2"/>
    <path d="${best}" fill="none" stroke="var(--series-1)" stroke-width="3" stroke-linejoin="round"/>
    ${dots}
    <circle cx="${bx}" cy="${by}" r="8" fill="var(--series-1)" stroke="var(--panel)" stroke-width="2"/>
    <text x="${Math.min(bx + 12, W - 4)}" y="${by - 12}" fill="var(--ink)" font-size="12" font-weight="600"
      text-anchor="${bx + 90 > W ? "end" : "start"}">best ${fmt(bestRow.score)} (${bestRow.version})</text>
    <text x="${W-m.r+6}" y="${y(last.run_best)+4}" fill="var(--ink-2)" font-size="11">${fmt(last.run_best,3)}</text>
    <line id="xh" y1="${m.t}" y2="${H-m.b}" stroke="var(--ink-3)" stroke-width="1" visibility="hidden"/>
    <rect id="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>
  </svg>`;

  const svg = $("chart").querySelector("svg"), tip = $("tip"), xh = $("xh");
  const show = ev => {
    const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * W / r.width;
    const i = Math.max(0, Math.min(n-1, Math.round(n === 1 ? 0 : (sx - m.l) / iw * (n-1)))), d = DATA[i];
    xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i)); xh.setAttribute("visibility", "visible");
    tip.innerHTML = `<b>#${i+1} · ${d.version}</b> <span class="m">+${ago(d.t - t0)}</span><br>` +
      `score <b>${fmt(d.score)}</b><br>running best <b>${fmt(d.run_best)}</b> <span class="m">(${d.run_best_v})</span>`;
    tip.style.display = "block";
    const px = x(i) / W * r.width, cw = $("chart").parentElement.clientWidth;
    tip.style.left = (px + 16 + tip.offsetWidth > cw ? px - tip.offsetWidth - 4 : px + 16) + "px";
    tip.style.top = (y(d.score) / H * r.height + 20) + "px";
  };
  svg.addEventListener("pointermove", show);
  svg.addEventListener("pointerleave", () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); });

  const insts = Object.keys(DATA[0].per_instance || {}).sort();
  $("table").innerHTML = `<table><tr><th>#</th><th>version</th><th>elapsed</th><th>score</th><th>best</th>${insts.map(k=>`<th>${k}</th>`).join("")}</tr>` +
    DATA.map((d,i) => `<tr><td>${i+1}</td><td>${d.version}</td><td>${ago(d.t - t0)}</td><td>${fmt(d.score)}</td><td>${fmt(d.run_best)}</td>` +
      insts.map(k => `<td>${fmt((d.per_instance||{})[k], 3)}</td>`).join("") + "</tr>").join("") + "</table>";
}
</script></body></html>"""


def render(rows, out, refresh=None):
    html = PAGE.replace("__DATA__", json.dumps(rows)).replace(
        "__REFRESH__", f'<meta http-equiv="refresh" content="{refresh}">' if refresh else "")
    out.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(HISTORY))
    ap.add_argument("--out", default=str(WORK / "progress.html"))
    ap.add_argument("--all", action="store_true", help="plot every run in history, not just the latest")
    ap.add_argument("--watch", type=float, default=0, help="re-render every N seconds")
    ap.add_argument("--backfill", action="store_true", help="score work/v*.py without a history entry first")
    ap.add_argument("--tl", type=float, default=20); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    hist, out = Path(a.history), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.backfill:
        backfill(a.tl, a.jobs)
    while True:
        rows = select(load(hist), a.all)
        render(rows, out, refresh=max(2, int(a.watch)) if a.watch else None)
        print(f"{len(rows)} evals -> {out}" + (f"  (best {max(r['score'] for r in rows):.4f})" if rows else ""))
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
