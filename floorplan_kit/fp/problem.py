"""Simplified floorplan problem (student edition): generator, checker, wirelength, legalizer.

Instance JSON
  outline : [W, H]                                 chip area
  blocks  : [{"w": .., "h": ..}, ...]              modules to place
  nets    : [{"a": i, "b": j, "w": weight}, ...]   2-pin connections between modules
  variant : {"forbidden": [[x,y,w,h],...], "edge": [i,...], "fixed": {"i": [x,y]},
             "distance": "manhattan"|"euclidean"|"squared", "rotate": false}
Solution   [[x, y], ...] lower-left corners   (or [x, y, r] with r in {0,1} when rotate is allowed)
Objective  wirelength = sum_w * dist(center_a, center_b)    (lower is better)
"""
import json, math, random

import numpy as np

EPS = 1e-6
DEFAULT_VARIANT = {"forbidden": [], "edge": [], "fixed": {}, "distance": "manhattan", "rotate": False}


# ----------------------------------------------------------------------------- generator
def _shelf_height(dims, W):
    x = y = row = 0.0
    for w, h in sorted(dims, key=lambda d: -d[1]):
        if x + w > W:
            x, y, row = 0.0, y + row, 0.0
        x += w; row = max(row, h)
    return y + row


def generate(n, seed, whitespace=0.15, aspect_range=(0.4, 2.5), size_sigma=0.8,
             group=8, intra=0.8, links_per_block=1.5, variant=None):
    rng = random.Random(seed)
    blocks = []
    for _ in range(n):
        a = rng.lognormvariate(0, size_sigma) * 100
        r = rng.uniform(*aspect_range)
        w = max(2, round((a * r) ** 0.5)); h = max(2, round(a / w))
        blocks.append({"w": w, "h": h})
    area = sum(b["w"] * b["h"] for b in blocks) / (1 - whitespace)
    W = max(round(area ** 0.5), max(b["w"] for b in blocks))
    H = max(round(area / W), math.ceil(_shelf_height([(b["w"], b["h"]) for b in blocks], W)))
    groups = max(1, n // group)
    gid = [rng.randrange(groups) for _ in range(n)]
    members = [[i for i in range(n) if gid[i] == g] for g in range(groups)]
    links = {}
    for _ in range(int(n * links_per_block)):
        a = rng.randrange(n)
        pool = members[gid[a]] if rng.random() < intra and len(members[gid[a]]) > 1 else range(n)
        b = rng.choice(list(pool))
        if a != b:
            k = (min(a, b), max(a, b)); links[k] = links.get(k, 0) + rng.choice([1, 1, 2, 4])
    nets = [{"a": a, "b": b, "w": w} for (a, b), w in sorted(links.items())]
    return {"name": f"syn{n}_s{seed}", "outline": [W, H], "blocks": blocks, "nets": nets,
            "variant": dict(DEFAULT_VARIANT, **(variant or {}))}


# ----------------------------------------------------------------------------- geometry
def variant(inst):
    return dict(DEFAULT_VARIANT, **inst.get("variant", {}))


def dims(inst, sol=None):
    w = np.array([b["w"] for b in inst["blocks"]], float)
    h = np.array([b["h"] for b in inst["blocks"]], float)
    if sol is not None and variant(inst)["rotate"]:
        r = np.array([int(p[2]) if len(p) > 2 else 0 for p in sol], bool)
        w, h = np.where(r, h, w), np.where(r, w, h)
    return w, h


def _xy(sol):
    a = np.array([[float(p[0]), float(p[1])] for p in sol], float)
    return a[:, 0], a[:, 1]


def check(inst, sol):
    """Return (ok, message). Hard rules: inside outline, no overlap, plus active variant rules."""
    n = len(inst["blocks"]); W, H = inst["outline"]; v = variant(inst)
    if not isinstance(sol, (list, tuple)) or len(sol) != n:
        return False, f"expected {n} placements"
    try:
        x, y = _xy(sol)
        if not v["rotate"] and any(len(p) > 2 and int(p[2]) for p in sol):
            return False, "rotation not allowed in this variant"
    except Exception as e:
        return False, f"bad entry: {e}"
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return False, "non-finite coordinate"
    w, h = dims(inst, sol)
    bad = np.where((x < -EPS) | (y < -EPS) | (x + w > W + EPS) | (y + h > H + EPS))[0]
    if len(bad):
        return False, f"block {bad[0]} outside outline"
    ox = np.minimum(x[:, None] + w[:, None], x[None] + w[None]) - np.maximum(x[:, None], x[None])
    oy = np.minimum(y[:, None] + h[:, None], y[None] + h[None]) - np.maximum(y[:, None], y[None])
    hit = (ox > EPS) & (oy > EPS); np.fill_diagonal(hit, False)
    if hit.any():
        i, j = np.argwhere(hit)[0]
        return False, f"blocks {i} and {j} overlap"
    for k, (fx, fy, fw, fh) in enumerate(v["forbidden"]):
        o = (np.minimum(x + w, fx + fw) - np.maximum(x, fx) > EPS) & (np.minimum(y + h, fy + fh) - np.maximum(y, fy) > EPS)
        if o.any():
            return False, f"block {np.argmax(o)} overlaps forbidden area {k}"
    for i, (fx, fy) in v["fixed"].items():
        i = int(i)
        if abs(x[i] - fx) > EPS or abs(y[i] - fy) > EPS:
            return False, f"fixed block {i} moved"
    for i in v["edge"]:
        if min(x[i], y[i], W - x[i] - w[i], H - y[i] - h[i]) > EPS:
            return False, f"edge block {i} not touching the outline"
    return True, "ok"


def wirelength(inst, sol):
    x, y = _xy(sol); w, h = dims(inst, sol)
    cx, cy = x + w / 2, y + h / 2
    a = np.array([e["a"] for e in inst["nets"]]); b = np.array([e["b"] for e in inst["nets"]])
    wt = np.array([e["w"] for e in inst["nets"]], float)
    dx, dy = np.abs(cx[a] - cx[b]), np.abs(cy[a] - cy[b])
    d = {"manhattan": dx + dy, "euclidean": np.hypot(dx, dy), "squared": dx ** 2 + dy ** 2}[variant(inst)["distance"]]
    return float((wt * d).sum())


# ----------------------------------------------------------------------------- legalizer
def legalize(inst, sol, k_near=40):
    """Return a legal placement close to `sol` (unchanged if already legal), or None on failure.
    Tries: greedy by decreasing area -> greedy bottom-up sweep -> shelf packing (base variant)."""
    if check(inst, sol)[0]:
        return [list(p) for p in sol]
    for order in ("area", "sweep"):
        out = _greedy(inst, sol, k_near, order)
        if out is not None:
            return out
    return _shelf_fallback(inst, sol)


def _shelf_fallback(inst, sol):
    """Rows sorted by height, blocks kept in their desired x order inside each row (base variant only)."""
    v = variant(inst)
    if v["forbidden"] or v["fixed"] or v["edge"]:
        return None
    W, H = inst["outline"]; w, h = dims(inst, sol); x0, y0 = _xy(sol)
    order = sorted(range(len(w)), key=lambda i: -h[i])        # same order the generator used -> always fits
    out = [None] * len(w); x = y = row = 0.0
    rot = [[int(p[2])] if len(p) > 2 and v["rotate"] else ([0] if v["rotate"] else []) for p in sol]
    for i in order:
        if x + w[i] > W + EPS:
            x, y, row = 0.0, y + row, 0.0
        out[i] = [x, y] + rot[i]; x += w[i]; row = max(row, h[i])
    return out if check(inst, out)[0] else None


def _greedy(inst, sol, k_near, mode):
    """Fixed blocks first, then edge blocks, then the rest (by decreasing area, or bottom-up sweep);
    each block goes to the nearest legal candidate generated from corners of nearby placed rectangles."""
    W, H = inst["outline"]; v = variant(inst); n = len(inst["blocks"])
    x0, y0 = _xy(sol); w, h = dims(inst, sol)
    rot = [int(p[2]) if len(p) > 2 and v["rotate"] else 0 for p in sol]
    dx = np.clip(np.nan_to_num(x0), 0, W - w); dy = np.clip(np.nan_to_num(y0), 0, H - h)
    rects = [list(r[:2]) + [r[0] + r[2], r[1] + r[3]] for r in v["forbidden"]]   # x0,y0,x1,y1
    out = [None] * n
    fixed = {int(i): xy for i, xy in v["fixed"].items()}
    edge = set(int(i) for i in v["edge"])
    for i, (fx, fy) in fixed.items():
        out[i] = [fx, fy] + ([rot[i]] if v["rotate"] else []); rects.append([fx, fy, fx + w[i], fy + h[i]])
    order = [i for i in sorted(edge) if i not in fixed] + \
            sorted((i for i in range(n) if i not in fixed and i not in edge),
                   key=(lambda i: -w[i] * h[i]) if mode == "area" else (lambda i: (dy[i], dx[i])))
    for i in order:
        wi, hi, px, py = w[i], h[i], dx[i], dy[i]
        R = np.array(rects, float).reshape(-1, 4)

        def cands(sub):
            c = [(px, py), (0, py), (W - wi, py), (px, 0), (px, H - hi), (0, 0), (W - wi, 0), (0, H - hi), (W - wi, H - hi)]
            for ax0, ay0, ax1, ay1 in sub:
                c += [(ax1, py), (ax0 - wi, py), (px, ay1), (px, ay0 - hi), (ax1, ay0), (ax1, ay1 - hi),
                      (ax0 - wi, ay0), (ax0 - wi, ay1 - hi), (ax0, ay1), (ax1 - wi, ay1), (ax0, ay0 - hi), (ax1 - wi, ay0 - hi)]
            sub2 = sub[:12]                               # corner combinations from the closest rects
            xs = np.r_[0, W - wi, px, sub2[:, 2], sub2[:, 0] - wi] if len(sub2) else np.array([0, W - wi, px])
            ys = np.r_[0, H - hi, py, sub2[:, 3], sub2[:, 1] - hi] if len(sub2) else np.array([0, H - hi, py])
            gx, gy = np.meshgrid(xs, ys)
            c = np.r_[np.array(c, float).reshape(-1, 2), np.c_[gx.ravel(), gy.ravel()]]
            c = np.clip(c, 0, [W - wi, H - hi])
            if i in edge:   # project every candidate onto each of the four edges
                c = np.concatenate([np.c_[np.zeros(len(c)), c[:, 1]], np.c_[np.full(len(c), W - wi), c[:, 1]],
                                    np.c_[c[:, 0], np.zeros(len(c))], np.c_[c[:, 0], np.full(len(c), H - hi)]])
            return c

        def best(c):
            if len(R):
                ok = ~(((c[:, None, 0] < R[None, :, 2] - EPS) & (c[:, None, 0] + wi > R[None, :, 0] + EPS) &
                        (c[:, None, 1] < R[None, :, 3] - EPS) & (c[:, None, 1] + hi > R[None, :, 1] + EPS)).any(1))
                c = c[ok]
            if not len(c):
                return None
            return c[np.argmin(np.abs(c[:, 0] - px) + np.abs(c[:, 1] - py))]

        near = R[np.argsort(np.abs((R[:, 0] + R[:, 2]) / 2 - px - wi / 2) + np.abs((R[:, 1] + R[:, 3]) / 2 - py - hi / 2))[:k_near]] if len(R) else R
        p = best(cands(near))
        if p is None and len(R) > k_near:
            p = best(cands(R))
        if p is None:                                   # dense grid scan as last resort
            step = max(min(wi, hi) / 2, 1e-3)
            gx, gy = np.meshgrid(np.arange(0, W - wi + EPS, step), np.arange(0, H - hi + EPS, step))
            p = best(np.c_[gx.ravel(), gy.ravel()])
        if p is None:
            return None
        out[i] = [float(p[0]), float(p[1])] + ([rot[i]] if v["rotate"] else [])
        rects.append([p[0], p[1], p[0] + wi, p[1] + hi])
    return out if check(inst, out)[0] else None


def displacement(sol_a, sol_b, inst):
    """Mean displacement between two placements, relative to sqrt(outline area)."""
    xa, ya = _xy(sol_a); xb, yb = _xy(sol_b)
    W, H = inst["outline"]
    return float(np.mean(np.abs(xa - xb) + np.abs(ya - yb)) / math.sqrt(W * H))


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="generate synthetic instances")
    ap.add_argument("--out", default="instances")
    ap.add_argument("--practice", action="store_true", help="only the small practice set")
    a = ap.parse_args()
    plan = {"dev": [(n, s) for n in (20, 50, 100, 200) for s in range(3)],
            "hidden": [(n, 1000 + s) for n in (20, 50, 100, 200) for s in range(3)]}
    if a.practice:
        plan = {"dev": [(n, s) for n in (20, 50, 100) for s in range(2)]}
    for split, items in plan.items():
        os.makedirs(f"{a.out}/{split}", exist_ok=True)
        for n, s in items:
            inst = generate(n, s)
            json.dump(inst, open(f"{a.out}/{split}/{inst['name']}.json", "w"))
    print("wrote", a.out)
