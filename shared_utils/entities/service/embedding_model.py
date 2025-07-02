# shared_utils/entities/service/embedding_model.py
import threading, torch
from functools import lru_cache
from sentence_transformers import SentenceTransformer

_LOCK = threading.Lock()

@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Load once per worker, without meta-tensor mode
    and without the invalid `device_map` kwarg.
    """
    with _LOCK:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L12-v2",
            device=device,              # ✅ correct way
            trust_remote_code=True,
            model_kwargs={"low_cpu_mem_usage": False}
        )

def _embed_sync(texts):
    model = _load_model()
    with torch.inference_mode():
        # encode() → np.ndarray(shape=(len(texts), dim))
        return model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64
        ).tolist()        # list[list[float]]