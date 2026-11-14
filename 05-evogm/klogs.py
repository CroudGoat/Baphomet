# Live Kaggle kernel log via kagglesdk logs stream (prints what arrives within the time limit)
import sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import *
api = KaggleApi(); api.authenticate()
buf = b""
with api.build_kaggle_client() as k:
    c = k.kernels.kernels_api_client
    rq = ApiGetKernelSessionLogsStreamRequest(); rq.user_name = 'cloudgoat'; rq.kernel_slug = sys.argv[1]
    try: rq.wait_for_logs_url_seconds = 20
    except Exception as e: print('wfl', e)
    try:
        r = c.get_kernel_session_logs_stream(rq)
        t0 = time.time()
        try:
            for ch in r.iter_content(4096):
                buf += ch
                if time.time() - t0 > float(sys.argv[2]): break
        except Exception as e:
            print("stream end:", repr(e)[:200])
    except Exception as e:
        print("err", repr(e)[:300])
open("/tmp/klive.txt", "wb").write(buf)
print(len(buf))
