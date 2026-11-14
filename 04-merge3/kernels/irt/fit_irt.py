# Kernel I (CPU): fit the MERGE^3 / tinyBenchmarks IRT model on Open LLM Leaderboard v2 per-item responses.
# py-irt multidim_2pl from Polo et al.'s modified py-irt (github.com/felipemaiapolo/py-irt, as used by tinyBenchmarks/MERGE^3), hierarchical priors, D=15 (MERGE^3 App. B.1), lr 0.1, epochs 2000, seed 42, deterministic
# (tinyBenchmarks tutorials/irt.py train_irt_model command).
import os, sys, json, glob, subprocess, time
import numpy as np
t0 = time.time()
def sh(c, t=None):
    print("$", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-6000:], r.stderr[-6000:], "rc", r.returncode, flush=True); return r
DATA = os.path.dirname(glob.glob("/kaggle/input/**/lb_responses.npz", recursive=True)[0])
sh("nproc; free -g; nvidia-smi -L 2>/dev/null; pip install -q pyro-ppl typer rich ordered-set toml jsonlines 2>&1 | tail -2", t=1800)
# Polo et al. py-irt fork (felipemaiapolo/py-irt @ dc3d2a09) shipped in the dataset (as a dir, or as a tarball if Kaggle kept it packed)
cli = glob.glob("/kaggle/input/**/polo_py_irt/py_irt/cli.py", recursive=True)
if not cli:
    tb = glob.glob("/kaggle/input/**/polo_py_irt.tar.gz", recursive=True)[0]
    sh(f"tar xzf {tb} -C /tmp"); cli = ["/tmp/polo_py_irt/py_irt/cli.py"]
PYIRT = os.path.dirname(os.path.dirname(cli[0]))
print("PYIRT", PYIRT, "models", len(np.load(f"{DATA}/lb_responses.npz")["names"]), flush=True)
import jsonlines
d = np.load(f"{DATA}/lb_responses.npz")
Y = d["Y"]; print("Y", Y.shape, flush=True)
D = int(os.environ.get("IRT_D", "15")); EPOCHS = int(os.environ.get("IRT_EPOCHS", "2000"))
rows = []
for i in range(Y.shape[0]):
    rows.append({"subject_id": str(i), "responses": {f"q{j}": int(Y[i, j]) for j in range(Y.shape[1]) if not np.isnan(Y[i, j])}})
with jsonlines.open("/kaggle/working/irt_dataset.jsonlines", "w") as w: w.write_all(rows)
dev = "cuda" if subprocess.run("nvidia-smi -L", shell=True, capture_output=True).returncode == 0 else "cpu"
sh(f"cd /tmp && PYTHONPATH={PYIRT} python -c 'from py_irt.cli import app; app()' train 'multidim_2pl' /kaggle/working/irt_dataset.jsonlines /kaggle/working/irt_model/ --dims {D} --lr 0.1 --epochs {EPOCHS} "
   f"--device {dev} --priors 'hierarchical' --seed 42 --deterministic --log-every 200", t=11 * 3600)
p = json.load(open("/kaggle/working/irt_model/best_parameters.json"))
print(p.keys(), flush=True)
# item ids are 'q{j}' -> map back to columns (py-irt reorders items: use item_ids mapping)
iid = p.get("item_ids"); sid = p.get("subject_ids")
disc = np.array(p["disc"]); diff = np.array(p["diff"]); abil = np.array(p["ability"])
# py-irt stores {index: item_id}; reorder rows so row j <-> column q{j}
order = np.zeros(Y.shape[1], dtype=int)
for k, v in (iid.items() if isinstance(iid, dict) else enumerate(iid)): order[int(v[1:])] = int(k)
disc, diff = disc[order], diff[order]
sorder = np.zeros(Y.shape[0], dtype=int)
for k, v in (sid.items() if isinstance(sid, dict) else enumerate(sid)): sorder[int(v)] = int(k)
abil = abil[sorder]
A = disc.T[None, :, :]; B = diff.T[None, :, :]
np.savez_compressed("/kaggle/working/irt_params.npz", A=A, B=B, ability=abil, names=d["names"], scen_mmlu_pro=d["scen_mmlu_pro"], scen_gpqa=d["scen_gpqa"])
print("A", A.shape, "B", B.shape, "secs", time.time() - t0)
os.remove("/kaggle/working/irt_dataset.jsonlines")
