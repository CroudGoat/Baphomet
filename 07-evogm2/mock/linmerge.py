# mock linmerge for the EvoGM v2 harness (no torch; downloads/extraction are sleeps + tiny files)
import os, json, time, random
BASE_ID = "Qwen/Qwen3.5-4B"
DONORS = [(s, "r/" + s, s == "neohorse") for s in ["agents_a1", "dsv4pro", "q38distill", "qwopus_coder", "qwopus_v3", "arex_turbo", "neohorse", "tmax"]]
SHORT = [d[0] for d in DONORS]; N = 8; GROUPS = ("attn", "mlp")
TS = float(os.environ.get("MOCK_TS", "0.002"))
def log(*a): print(*a)
def group_of(k): return 1 if ".mlp." in k else 0
def expand(l): return [float(v) for v in l] * 2
def download(repo, local_dir, patterns=None):
    time.sleep(random.uniform(0.5, 1.5) * 300 * TS)
    os.makedirs(local_dir, exist_ok=True); open(local_dir + "/w.safetensors", "wb").write(b"x" * 1000)
def extract_tv(b, d, remap, out, device="cuda"):
    time.sleep(random.uniform(0.5, 1.0) * 120 * TS); open(out, "wb").write(b"t"); return dict(n=1, zero=0, missing=0, rel_qerr=0.02)
