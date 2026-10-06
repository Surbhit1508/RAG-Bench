"""Fetch datasets/models from the public internet via `curl`.

Why not just use `requests`/`huggingface_hub`'s own downloader? Because
Walmart's corporate proxy requires NTLM authentication, which Windows
`curl` negotiates automatically using your logged-in credentials (SSPI),
but Python's httpx/requests stacks do not -- they fail with a 407.
Rather than wrestling NTLM auth into every HTTP library we depend on, we
shell out to `curl` for the handful of downloads we need, and everything
lands in a plain local folder that `transformers`/`sentence-transformers`
can load directly via `from_pretrained(local_dir)`. No HF cache internals
required, no proxy fighting inside three different libraries.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ragbench import config

# HuggingFace's own API is reachable directly, but the actual weight files
# redirect to an S3/Xet CDN (us.aws.cdn.hf.co) our corp proxy blocks with a
# 407. Walmart's Artifactory has a native HuggingFaceML remote repo that
# fetches those files server-side and hands them back over a domain we
# CAN reach -- so metadata comes from HF directly, file bytes come via
# Artifactory.
HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://generic.ci.artifacts.walmart.com/artifactory/api/huggingfaceml/hub-hf-release-remote"
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

MODELS_DIR = config.PROJECT_ROOT / "models"

# We only need plain PyTorch/safetensors + tokenizer artifacts to load a
# model -- not alternate framework exports (ONNX/OpenVINO/TF/Flax/coreml),
# training scripts, or repo docs. An allowlist is more robust here than a
# blocklist: new repos keep adding new export formats we've never seen,
# but the set of files actually needed to `from_pretrained()` is stable.
_ALLOWED_SUFFIXES = (".json", ".safetensors", ".bin", ".txt", ".model")
_SKIP_DIR_MARKERS = ("onnx/", "openvino/", "tf_model", "flax_model", "rust_model", "coreml/")


def _should_fetch(fname: str) -> bool:
    lower = fname.lower()
    if not lower.endswith(_ALLOWED_SUFFIXES):
        return False
    if any(marker in lower for marker in _SKIP_DIR_MARKERS):
        return False
    return True


def _proxy_env() -> dict:
    env = os.environ.copy()
    env.setdefault("HTTP_PROXY", "http://sysproxy.wal-mart.com:8080")
    env.setdefault("HTTPS_PROXY", "http://sysproxy.wal-mart.com:8080")
    return env


def _curl(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-sL", "-f", "-o", str(out_path), url],
        env=_proxy_env(),
    )
    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"curl failed ({result.returncode}) fetching {url}")


def fetch_squad_dev(out_path: Path | None = None) -> Path:
    out_path = out_path or (config.DATA_DIR / "squad_dev_v1.1.json")
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    print(f"Fetching SQuAD dev set -> {out_path}")
    _curl(SQUAD_DEV_URL, out_path)
    return out_path


def local_model_dir(repo_id: str) -> Path:
    """Path a repo's files land in -- guarantees the fetch has happened."""
    return fetch_hf_repo(repo_id)


def fetch_hf_repo(repo_id: str) -> Path:
    """Download every file a HF model repo needs into a local folder.

    Idempotent: a `.fetched_ok` marker means we skip re-downloading on
    subsequent runs (important -- these models are hundreds of MB+ and
    we don't want every script invocation to re-pull them).
    """
    local_dir = MODELS_DIR / repo_id.replace("/", "__")
    marker = local_dir / ".fetched_ok"
    if marker.exists():
        return local_dir

    meta_path = local_dir / "_meta.json"
    print(f"Fetching model repo listing for {repo_id} ...")
    _curl(f"{HF_API}/{repo_id}", meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    siblings = meta.get("siblings", [])
    if not siblings:
        raise RuntimeError(f"No files found for {repo_id}. API response: {meta}")

    for sib in siblings:
        fname = sib["rfilename"]
        if not _should_fetch(fname):
            continue
        dest = local_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"  fetching {repo_id}/{fname} ...")
        try:
            _curl(f"{HF_RESOLVE}/{repo_id}/resolve/main/{fname}", dest)
        except RuntimeError as exc:
            print(f"    WARNING: skipping {fname} ({exc})")

    marker.write_text("ok", encoding="utf-8")
    meta_path.unlink(missing_ok=True)
    return local_dir


if __name__ == "__main__":
    fetch_squad_dev()
