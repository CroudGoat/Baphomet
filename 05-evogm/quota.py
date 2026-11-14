from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types import kernels_api_service as T
api = KaggleApi(); api.authenticate()
names=[n for n in dir(T) if 'Quota' in n]; print(names)
with api.build_kaggle_client() as k:
    c=k.kernels.kernels_api_client
    rq=getattr(T,[n for n in names if n.endswith('Request')][0])()
    r=c.get_accelerator_quota_statistics(rq)
    print(r.to_dict() if hasattr(r,'to_dict') else r)
