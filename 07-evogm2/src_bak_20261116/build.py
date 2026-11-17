import base64, json, os, subprocess, sys
D = "/workspace/work/evogm2"
src = "".join(open(f"{D}/src/{p}").read() for p in ("part_a.py", "part_b.py", "part_c.py"))
w = base64.b64encode(open(f"{D}/src/gpuworker4.py", "rb").read()).decode()
assert src.count("__WORKER_B64__") == 1
out = src.replace("__WORKER_B64__", w)
os.makedirs(f"{D}/kernel_main", exist_ok=True)
open(f"{D}/kernel_main/evogm2_main.py", "w").write(out)
open(f"{D}/src/evogm2_src.py", "w").write(src)   # readable (worker not embedded)
compile(out, "evogm2_main.py", "exec"); compile(open(f"{D}/src/gpuworker4.py").read(), "gpuworker4.py", "exec")
json.dump({"id": "cloudgoat/evogm2-main", "title": "evogm2-main", "code_file": "evogm2_main.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["cloudgoat/evogm2-data"], "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(f"{D}/kernel_main/kernel-metadata.json", "w"), indent=1)
print("built", len(out), "bytes")
