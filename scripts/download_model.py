import json
import os
from pathlib import Path
import re
import shutil


def valid_cache(target, repo, revision):
    try:
        manifest = json.loads((target / "complete.json").read_text())
        if manifest["repo"] != repo or manifest["revision"] != revision or not manifest["files"]:
            return False
        for entry in manifest["files"]:
            file = target / entry["path"]
            if not file.resolve().is_relative_to(target.resolve()) or not file.is_file() or file.stat().st_size != entry["size"]:
                return False
        return (target / "config.json").is_file() and any(target.glob("*.safetensors"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def download(root, repo, revision, token=None, snapshot=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use an immutable 40-character HF commit")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    import fcntl
    with (root / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / revision
        if valid_cache(target, repo, revision):
            print('{"event":"model_cache_hit"}', flush=True)
            return target
        staging = root / (revision + ".partial")
        if snapshot is None:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download
        snapshot(repo_id=repo, revision=revision, local_dir=str(staging), token=token,
                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
                 ignore_patterns=["pytorch_model*", "*.bin", "*.pt", "*.h5"])
        if not (staging / "config.json").is_file() or not any(staging.glob("*.safetensors")):
            raise ValueError("Required model files missing")

        shutil.rmtree(staging / ".cache", ignore_errors=True)
        files = [{"path": str(p.relative_to(staging)), "size": p.stat().st_size}
                 for p in sorted(staging.rglob("*")) if p.is_file() and p.name != "complete.json"]
        (staging / "complete.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": files}))
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        print('{"event":"model_download_complete"}', flush=True)
        return target


if __name__ == "__main__":
    try:
        download(os.getenv("MODEL_ROOT", "/models"), os.environ["MODEL_ID"], os.environ["MODEL_REVISION"], os.getenv("HF_TOKEN") or None)
    except Exception:

        print('{"event":"model_download_failed"}', flush=True)
        raise SystemExit(1)
