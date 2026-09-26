"""Load architecture-specific checkpoints, including the original baseline."""
import math
import torch


def components(architecture):
    if architecture == "baseline_vit":
        from baseline_vit.config import load_config
        from baseline_vit.model import BaselineViT
        return load_config, BaselineViT, "baseline_vit_v1"
    if architecture == "yalamanchi_vit":
        from yalamanchi_vit.config import load_config
        from yalamanchi_vit.model import YalamanchiViT, ARCHITECTURE
        return load_config, YalamanchiViT, ARCHITECTURE
    raise ValueError("Unknown architecture")


def read_checkpoint(path, architecture):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    _, _, expected = components(architecture)
    legacy = (architecture == "baseline_vit" and "architecture" not in saved
              and "depth" in saved.get("config", {}).get("model", {}))
    if not legacy and (saved.get("architecture") != expected or saved.get("format_version") != 1):
        raise ValueError("Checkpoint is not a supported checkpoint for this architecture")
    return saved


def check_cache(saved, index):
    if saved["cache_fingerprint"] != index.fingerprint:
        raise ValueError("Cache metadata changed since training")
    if not math.isclose(saved["dt_reference"], index.dt_reference, rel_tol=1e-12, abs_tol=0):
        raise ValueError("Reference timestep changed since training")


def restore_model(saved, device, architecture):
    _, model_class, _ = components(architecture)
    model = model_class(**saved["config"]["model"]).to(device)
    model.load_state_dict(saved["model"])
    return model
