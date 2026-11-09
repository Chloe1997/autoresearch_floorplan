#!/usr/bin/env python3
"""Convert ICCAD04 (TILOS MacroPlacement / Circuit-Training clustered netlist) testcases
into the hackathon's simplified floorplan JSON. Pure stdlib, works offline.

Input : <dir>/netlist.pb.txt  (+ <dir>/initial.plc for the canvas size)
        e.g. external/MacroPlacement/Testcases/ICCAD04/ibm01
Output: {"name", "source", "outline":[W,H], "blocks":[{"w","h"}], "nets":[{"a","b","w"}],
         "variant":{...defaults...}, "meta":{...}}

Modes
  --blocks hard  : only hard macros become blocks (default, ~250-540 blocks)
  --blocks all   : hard macros + soft macros (std-cell clusters) become blocks (~1-3k blocks)
  In "hard" mode, connectivity that passes through soft macros is folded into
  hard-to-hard links (hard -> soft cluster -> hard), so most of the real structure survives.

Multi-pin nets (k distinct endpoints, k <= --max-degree) become a clique of pairwise links,
each with weight w_net / (k-1); parallel links are merged by summing weights.
Ports (I/O pins) are dropped in the base format (kept only as counts in meta).

Derived instances (for hidden sets):  --perturb SEED  shuffles block ids and scales each
block's w/h by U(1-p, 1+p); --whitespace W re-sizes the outline to W free area (square-ish,
same aspect as the original canvas) and grows H if needed so a shelf packing fits.

Examples
  python convert_iccad04.py ICCAD04/ibm01 -o dev/ibm01.json
  python convert_iccad04.py ICCAD04/ibm07 -o hidden/ibm07_d3.json --perturb 3 --whitespace 0.15
  python convert_iccad04.py ICCAD04 --all -o out_dir/          # every ibmXX subfolder
"""
import argparse, json, math, os, random, re, sys
from collections import defaultdict
from itertools import combinations

NODE_RE = re.compile(r'^node \{\n(.*?)^\}', re.S | re.M)
NAME_RE = re.compile(r'^\s*name: "([^"]*)"', re.M)
INPUT_RE = re.compile(r'^\s*input: "([^"]*)"', re.M)
ATTR_RE = re.compile(r'attr \{\s*key: "([^"]+)"\s*value \{\s*(f|placeholder|i): ("?[^"\n}]*"?)\s*\}\s*\}', re.S)


def parse_pb(path):
    """Yield (name, inputs, attrs) for every node of a CT protobuf-text netlist."""
    text = open(path).read()
    for m in NODE_RE.finditer(text):
        body = m.group(1)
        nm = NAME_RE.search(body)
        attrs = {}
        for k, kind, v in ATTR_RE.findall(body):
            v = v.strip('"')
            attrs[k] = float(v) if kind == "f" else (int(v) if kind == "i" else v)
        yield (nm.group(1) if nm else None), INPUT_RE.findall(body), attrs


def canvas_from_plc(path):
    for line in open(path):
        m = re.match(r'#\s*Width\s*:\s*([\d.]+)\s+Height\s*:\s*([\d.]+)', line)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def shelf_height(blocks, W):
    x = y = row = 0.0
    for b in sorted(blocks, key=lambda b: -b["h"]):
        if x + b["w"] > W:
            x, y, row = 0.0, y + row, 0.0
        x += b["w"]; row = max(row, b["h"])
    return y + row


def convert(d, mode="hard", max_degree=10, scale=100.0, perturb=None, p=0.2, whitespace=None):
    pb = os.path.join(d, "netlist.pb.txt")
    name = os.path.basename(os.path.normpath(d))
    hard, soft, owner, ports = {}, {}, {}, set()
    drivers = []                                     # (driver_pin, [sink_pins], weight)
    for nm, inputs, a in parse_pb(pb):
        t = a.get("type")
        if t == "MACRO":
            hard[nm] = (a["width"], a["height"])
        elif t == "macro":
            soft[nm] = (a["width"], a["height"])
        elif t in ("MACRO_PIN", "macro_pin"):
            owner[nm] = a["macro_name"]
        elif t == "PORT":
            ports.add(nm); owner[nm] = nm
        if inputs:
            drivers.append((nm, inputs, float(a.get("weight", 1.0))))

    is_block = (lambda m: m in hard) if mode == "hard" else (lambda m: m in hard or m in soft)
    names = sorted(hard) + ([] if mode == "hard" else sorted(soft))
    idx = {m: i for i, m in enumerate(names)}

    # hyperedges at macro level (driver owner + sink owners)
    hyper = []
    for drv, sinks, w in drivers:
        ms = {owner.get(drv, drv)} | {owner.get(s, s) for s in sinks}
        hyper.append((ms, w))

    links = defaultdict(float)
    dropped_big = 0

    def add_clique(members, w):
        nonlocal dropped_big
        k = len(members)
        if k < 2:
            return
        if k > max_degree:
            dropped_big += 1; return
        for a, b in combinations(sorted(members), 2):
            links[(a, b)] += w / (k - 1)

    if mode == "all":
        for ms, w in hyper:
            add_clique({idx[m] for m in ms if is_block(m)}, w)
    else:
        # direct hard-hard connectivity
        soft_nbrs = defaultdict(lambda: defaultdict(float))   # soft cluster -> {hard idx: weight}
        for ms, w in hyper:
            hs = {idx[m] for m in ms if m in hard}
            add_clique(hs, w)
            for s in (m for m in ms if m in soft):
                for h in hs:
                    soft_nbrs[s][h] += w
        # fold one soft cluster hop: hard - soft - hard (half weight, cluster degree capped)
        for s, nb in soft_nbrs.items():
            hs = sorted(nb, key=lambda h: -nb[h])[:max_degree]
            if len(hs) >= 2:
                for a, b in combinations(sorted(hs), 2):
                    links[(a, b)] += 0.5 * min(nb[a], nb[b]) / (len(hs) - 1)

    sizes = [hard[m] if m in hard else soft[m] for m in names]
    blocks = [{"w": round(w * scale, 3), "h": round(h * scale, 3)} for w, h in sizes]
    canvas = canvas_from_plc(os.path.join(d, "initial.plc"))
    W, H = (canvas[0] * scale, canvas[1] * scale) if canvas else (None, None)

    perm = list(range(len(blocks)))
    if perturb is not None:
        rng = random.Random(perturb)
        rng.shuffle(perm)                                   # new id = position in perm
        inv = {old: new for new, old in enumerate(perm)}
        blocks = [{"w": round(blocks[o]["w"] * rng.uniform(1 - p, 1 + p), 3),
                   "h": round(blocks[o]["h"] * rng.uniform(1 - p, 1 + p), 3)} for o in perm]
        links = {tuple(sorted((inv[a], inv[b]))): w for (a, b), w in links.items()}
        name = f"{name}_d{perturb}"

    area = sum(b["w"] * b["h"] for b in blocks)
    if whitespace is not None or W is None:
        ws = 0.15 if whitespace is None else whitespace
        aspect = (W / H) if W else 1.0
        H = math.sqrt(area / (1 - ws) / aspect); W = H * aspect
    W = max(W, max(b["w"] for b in blocks))
    H = max(H, shelf_height(blocks, W))                      # guarantee a legal packing exists
    nets = [{"a": a, "b": b, "w": round(w, 4)} for (a, b), w in sorted(links.items()) if w > 0]

    return {
        "name": name,
        "source": "ICCAD04 " + os.path.basename(os.path.normpath(d)) +
                  " via TILOS MacroPlacement clustered netlist (partcleda/macro-place-challenge-2026)",
        "outline": [round(W, 3), round(H, 3)],
        "blocks": blocks,
        "nets": nets,
        "variant": {"forbidden": [], "edge": [], "fixed": {}, "distance": "manhattan", "rotate": False},
        "meta": {"mode": mode, "hard_macros": len(hard), "soft_macros": len(soft), "ports_dropped": len(ports),
                 "hypernets": len(hyper), "hypernets_over_max_degree": dropped_big, "links": len(nets),
                 "utilization": round(area / (W * H), 4), "scale": scale, "perturb": perturb},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="testcase dir (contains netlist.pb.txt) or, with --all, the ICCAD04 dir")
    ap.add_argument("-o", "--out", required=True, help="output .json (or directory with --all)")
    ap.add_argument("--all", action="store_true", help="convert every subfolder of src")
    ap.add_argument("--blocks", choices=["hard", "all"], default="hard")
    ap.add_argument("--max-degree", type=int, default=10)
    ap.add_argument("--scale", type=float, default=100.0, help="um -> output units")
    ap.add_argument("--perturb", type=int, default=None, help="seed for a derived instance")
    ap.add_argument("--perturb-range", type=float, default=0.2)
    ap.add_argument("--whitespace", type=float, default=None, help="resize outline to this free-area ratio")
    a = ap.parse_args()
    dirs = sorted(os.path.join(a.src, x) for x in os.listdir(a.src)
                  if os.path.isfile(os.path.join(a.src, x, "netlist.pb.txt"))) if a.all else [a.src]
    for d in dirs:
        inst = convert(d, a.blocks, a.max_degree, a.scale, a.perturb, a.perturb_range, a.whitespace)
        out = os.path.join(a.out, inst["name"] + ".json") if a.all else a.out
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(inst, open(out, "w"))
        m = inst["meta"]
        print(f'{inst["name"]:12s} blocks={len(inst["blocks"]):5d} links={m["links"]:6d} '
              f'util={m["utilization"]:.2f} outline={inst["outline"]} -> {out}', file=sys.stderr)


if __name__ == "__main__":
    main()
