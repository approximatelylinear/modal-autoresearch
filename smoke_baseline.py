"""Modal smoke test: reproduce hydra's frozen MiniLM baseline on scifact.

Goal: validate that hydra's environment + BEIR data path + eval code all work
on Modal, by reproducing the `0597b29` baseline row from results.tsv
(scifact NDCG@10 ≈ 0.6451).

Usage:
    pip install modal && modal token new
    modal run smoke_baseline.py

If the printed NDCG@10 matches the TSV (within ~1e-3), the substrate is good.
"""

from pathlib import Path

import modal

HYDRA_REPO = Path(__file__).parent.parent / "hydra"

# Build the image straight from hydra's locked deps. Using uv.lock as the
# source of truth captures whatever dep pinning was needed locally.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .add_local_file(
        str(HYDRA_REPO / "pyproject.toml"), "/work/pyproject.toml", copy=True
    )
    .add_local_file(str(HYDRA_REPO / "uv.lock"), "/work/uv.lock", copy=True)
    .add_local_file(str(HYDRA_REPO / "README.md"), "/work/README.md", copy=True)
    # Install hydra's locked deps into the *system* Python that Modal uses,
    # so we don't have to fight Modal's own interpreter / PATH. `--no-install-project`
    # skips building hydra itself; we mount its source separately below.
    .run_commands(
        "mkdir -p /work/hydra && touch /work/hydra/__init__.py",
        "cd /work && uv export --frozen --no-hashes --no-emit-project -o /tmp/reqs.txt",
        "uv pip install --system -r /tmp/reqs.txt",
    )
    # Mount the live hydra source last so code changes don't bust the dep layer.
    .add_local_dir(str(HYDRA_REPO / "hydra"), "/work/hydra")
)

app = modal.App("hydra-smoke", image=image)

# Persist BEIR downloads + HF model cache across runs.
beir_vol = modal.Volume.from_name("hydra-beir-cache", create_if_missing=True)
hf_vol = modal.Volume.from_name("hydra-hf-cache", create_if_missing=True)


@app.function(
    gpu="T4",
    timeout=60 * 30,
    volumes={"/cache/beir": beir_vol, "/cache/hf": hf_vol},
)
def baseline_scifact() -> dict:
    import hashlib
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    sys.path.insert(0, "/work")

    # Content-addressed hash of the resolved dep set. Two runs with the same
    # image_hash have byte-identical installed packages, regardless of whether
    # Modal rebuilt the underlying layer.
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    image_hash = hashlib.sha256(freeze.encode()).hexdigest()[:12]
    print(f"image_hash={image_hash} ({freeze.count(chr(10))} packages)", flush=True)

    import torch

    from hydra.data.beir_loader import load_beir_dataset
    from hydra.eval.evaluator import evaluate_baseline

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    ds = load_beir_dataset("scifact", data_dir="/cache/beir")
    print(
        f"scifact: {len(ds.corpus)} docs, {len(ds.queries)} queries", flush=True
    )

    result = evaluate_baseline("sentence-transformers/all-MiniLM-L6-v2", ds)
    print(f"result: {result}", flush=True)

    # Best-effort: extract metrics into a plain dict regardless of result type.
    if hasattr(result, "__dict__"):
        out = {k: v for k, v in vars(result).items() if not k.startswith("_")}
    elif isinstance(result, dict):
        out = result
    else:
        out = {"repr": repr(result)}

    out["image_hash"] = image_hash
    out["freeze"] = freeze

    beir_vol.commit()
    hf_vol.commit()
    return out


@app.local_entrypoint()
def main():
    out = baseline_scifact.remote()
    freeze = out.pop("freeze", "")
    print("\n=== smoke test result ===")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("\nExpected (results.tsv row 0597b29): scifact_ndcg10 ≈ 0.6451")

    # Persist the resolved dep set keyed by hash so we can diff future runs.
    if freeze:
        from pathlib import Path
        freeze_dir = Path(__file__).parent / "image_freezes"
        freeze_dir.mkdir(exist_ok=True)
        path = freeze_dir / f"{out['image_hash']}.txt"
        if not path.exists():
            path.write_text(freeze)
            print(f"\nWrote frozen deps -> {path}")
        else:
            print(f"\nimage_hash {out['image_hash']} already on disk")
