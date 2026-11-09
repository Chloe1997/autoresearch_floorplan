"""Run a solver (a .py file defining place(inst, time_limit)) on a split and score it.

Every output goes through the official legalizer first (a legal output is left unchanged).
  s_i  = ref_wirelength_i / your_wirelength_i       (failure / timeout / still illegal -> 0.1)
  SCORE = geometric mean of s_i over the split        (baseline = 1.00 by construction)

usage
  python fp/runner.py solvers/baseline_place.py                 # score on instances/dev
  python fp/runner.py solvers/baseline_place.py --make-ref      # organizer: write instances/ref.json
  python fp/runner.py my_place.py --split hidden --tl 20 --jobs 4
"""
import argparse, glob, json, math, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from problem import check, displacement, legalize, wirelength  # noqa: E402

FAIL = 0.1
CHILD = r'''
import json, sys, time, random, importlib.util
seed = int(sys.argv[5]); random.seed(seed)
try:
    import numpy as np; np.random.seed(seed)
except Exception: pass
try:
    import torch; torch.manual_seed(seed); torch.set_num_threads(int(sys.argv[6]))
except Exception: pass
spec = importlib.util.spec_from_file_location("solver", sys.argv[1]); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
inst = json.load(open(sys.argv[2])); t0 = time.time()
sol = m.place(inst, float(sys.argv[4]))
json.dump({"sol": [[float(v) for v in p] for p in sol], "sec": time.time() - t0}, open(sys.argv[3], "w"))
'''


def run_one(solver, inst_path, tl=20.0, seed=0, threads=2):
    solver, inst_path = os.path.abspath(solver), os.path.abspath(inst_path)
    inst = json.load(open(inst_path))
    res = {"inst": inst["name"], "legal": False, "wl": None, "msg": "", "sec": None, "disp": None}
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    with tempfile.TemporaryDirectory() as td:
        out, child = os.path.join(td, "out.json"), os.path.join(td, "child.py")
        open(child, "w").write(CHILD)
        try:
            p = subprocess.run([sys.executable, "-I", child, solver, inst_path, out, str(tl), str(seed), str(threads)],
                               capture_output=True, text=True, timeout=tl * 1.5 + 5, cwd=td, env=env)
        except subprocess.TimeoutExpired:
            res["msg"] = "timeout"; return res
        if p.returncode != 0 or not os.path.exists(out):
            res["msg"] = "crash: " + p.stderr.strip()[-600:]; return res
        r = json.load(open(out))
    res["sec"] = round(r["sec"], 2)
    raw_ok, raw_msg = check(inst, r["sol"])
    fixed = legalize(inst, r["sol"])
    if fixed is None:
        res["msg"] = "legalizer failed: " + raw_msg; return res
    res.update(legal=True, wl=wirelength(inst, fixed), disp=round(displacement(r["sol"], fixed, inst), 4),
               msg="ok" if raw_ok else "legalized (" + raw_msg + ")")
    return res


def score(solver, split="dev", tl=20.0, inst_dir=None, jobs=4, seed=0):
    inst_dir = Path(inst_dir or HERE.parent / "instances")
    refs = json.load(open(inst_dir / "ref.json"))
    files = sorted(glob.glob(str(inst_dir / split / "*.json")))
    with ThreadPoolExecutor(jobs) as ex:
        rows = list(ex.map(lambda f: run_one(solver, f, tl, seed), files))
    for r in rows:
        r["score"] = refs[r["inst"]] / r["wl"] if r["legal"] and r["wl"] > 0 else FAIL
    gm = math.exp(sum(math.log(r["score"]) for r in rows) / len(rows))
    return {"score": gm, "per_instance": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solver"); ap.add_argument("--split", default="dev")
    ap.add_argument("--tl", type=float, default=20); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dir", default=None, help="instances dir (default: ./instances)")
    ap.add_argument("--make-ref", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir or HERE.parent / "instances")
    if a.make_ref:
        ref = {}
        for split in sorted(p.name for p in d.iterdir() if p.is_dir()):
            files = sorted(glob.glob(str(d / split / "*.json")))
            with ThreadPoolExecutor(a.jobs) as ex:
                for r in ex.map(lambda f: run_one(a.solver, f, a.tl), files):
                    assert r["legal"], r
                    ref[r["inst"]] = r["wl"]; print(f'{split:7s} {r["inst"]:14s} wl={r["wl"]:.1f} sec={r["sec"]}')
        json.dump(ref, open(d / "ref.json", "w"), indent=1)
        print(f"ref.json: {len(ref)} instances"); return
    s = score(a.solver, a.split, a.tl, d, a.jobs)
    for r in s["per_instance"]:
        print(f'{r["inst"]:14s} score={r["score"]:.3f} sec={r["sec"]} disp={r["disp"]}  {r["msg"][:70]}')
    print(f'SCORE[{a.split}] = {s["score"]:.4f}')


if __name__ == "__main__":
    main()
