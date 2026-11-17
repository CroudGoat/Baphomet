import base64, json, os, sys
D = "/workspace/work/evogm-dfs"
src = "".join(open(f"{D}/src/{p}").read() for p in ("part_a.py", "part_b.py", "part_c.py"))
w = base64.b64encode(open(f"{D}/src/gpuworker_dfs.py", "rb").read()).decode()
l = base64.b64encode(open(f"{D}/src/dfslib.py", "rb").read()).decode()
assert src.count("__WORKER_B64__") == 1 and src.count("__DFSLIB_B64__") == 1
out = src.replace("__WORKER_B64__", w).replace("__DFSLIB_B64__", l)
os.makedirs(f"{D}/kernel", exist_ok=True)
open(f"{D}/kernel/evogm_dfs.py", "w").write(out)
open(f"{D}/src/evogm_dfs_src.py", "w").write(src)   # readable (worker / dfslib not embedded)
compile(out, "evogm_dfs.py", "exec"); compile(open(f"{D}/src/gpuworker_dfs.py").read(), "gpuworker_dfs.py", "exec")
json.dump({"id": "cloudgoat/evogm-dfs", "title": "evogm-dfs", "code_file": "evogm_dfs.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["cloudgoat/evogm2-data", "cloudgoat/evogm-dfs-data"], "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(f"{D}/kernel/kernel-metadata.json", "w"), indent=1)
print("built", len(out), "bytes")
