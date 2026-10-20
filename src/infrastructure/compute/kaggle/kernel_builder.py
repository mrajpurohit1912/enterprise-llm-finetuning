"""
src/infrastructure/compute/kaggle/kernel_builder.py
Generates Kaggle cloud worker execution packages (worker script and metadata).
"""

import base64
import io
import json
import logging
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import Optional

from src.domain.exceptions import RemoteComputeError

logger = logging.getLogger(__name__)


class KaggleKernelBuilder:
    """Builds and packages executable Kaggle kernel directories for remote execution."""

    def __init__(
        self,
        repo_url: str = "https://github.com/mrajpurohit1912/enterprise-llm-finetuning.git",
        default_branch: str = "main",
    ) -> None:
        self.repo_url = repo_url
        self.default_branch = default_branch

    @staticmethod
    def sanitize_slug(name: str) -> str:
        """Converts arbitrary experiment names to valid Kaggle kernel slugs (a-z, 0-9, hyphens)."""
        clean = re.sub(r"[^a-z0-9-]", "-", name.lower().replace("_", "-"))
        clean = re.sub(r"-+", "-", clean).strip("-")
        return clean[:45] if clean else "fine-tuning-job"

    @staticmethod
    def _create_src_tarball_base64() -> str:
        """Compresses the local 'src' directory into a base64-encoded gzipped tarball."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(
                "src",
                arcname="src",
                filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t,
            )
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def build_package(
        self,
        experiment_name: str,
        config_path: Path,
        output_dir: Path,
        username: str,
        git_branch: Optional[str] = None,
    ) -> Path:
        """
        Compiles the self-contained worker bootstrap script and kernel-metadata.json
        into the designated output directory ready for 'kaggle kernels push'.
        """
        if not username:
            raise RemoteComputeError(
                "Kaggle username is required to build kernel metadata. "
                "Ensure ~/.kaggle/kaggle.json or KAGGLE_USERNAME is set."
            )

        if not config_path.exists():
            raise RemoteComputeError(f"Configuration file not found: {config_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        slug = self.sanitize_slug(experiment_name)
        target_branch = git_branch or self.default_branch

        # 1. Read active configuration file, environment tokens, and bundle local src/ package
        config_yaml_content = config_path.read_text(encoding="utf-8")
        hf_token = os.getenv("HF_TOKEN")
        wandb_key = os.getenv("WANDB_API_KEY")
        src_payload = self._create_src_tarball_base64()

        # Also write the local copy for inspection
        bundled_config_path = output_dir / "finetuning_config.yaml"
        bundled_config_path.write_text(config_yaml_content, encoding="utf-8")

        # 2. Render self-contained worker.py bootstrap script
        worker_script_path = output_dir / "worker.py"
        worker_code = self._generate_worker_script(
            repo_url=self.repo_url,
            git_branch=target_branch,
            config_yaml_content=config_yaml_content,
            src_payload=src_payload,
            hf_token=hf_token,
            wandb_key=wandb_key,
        )
        worker_script_path.write_text(worker_code, encoding="utf-8")

        # 3. Generate kernel-metadata.json
        metadata = {
            "id": f"{username}/{slug}",
            "title": slug,
            "code_file": "worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }

        meta_file = output_dir / "kernel-metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Compiled Kaggle kernel deployment package for '%s/%s' at %s",
            username,
            slug,
            output_dir,
        )
        return output_dir

    def _generate_worker_script(
        self,
        repo_url: str,
        git_branch: str,
        config_yaml_content: str,
        src_payload: str,
        hf_token: Optional[str] = None,
        wandb_key: Optional[str] = None,
    ) -> str:
        """Generates self-contained python worker script executed on Kaggle Cloud GPUs."""
        escaped_yaml = repr(config_yaml_content)
        escaped_src = repr(src_payload)
        escaped_hf_token = repr(hf_token or "")
        escaped_wandb_key = repr(wandb_key or "")

        return f'''"""
Automated Kaggle Cloud Execution Worker for Enterprise LLM Fine-Tuning.
Generated dynamically by KaggleKernelBuilder.
"""

import base64
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

print("=" * 70)
print(">>> [Kaggle Worker] Initializing Enterprise LLM Fine-Tuning Runtime...")
print("=" * 70)

# 1. Resolve Credentials (from Kaggle Secrets or runtime env)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    for secret in ["HF_TOKEN", "WANDB_API_KEY"]:
        try:
            val = user_secrets.get_secret(secret)
            if val:
                os.environ[secret] = val
                print(f">>> [Kaggle Worker] Loaded '{{secret}}' from Kaggle Secrets.")
        except Exception:
            pass
except Exception:
    pass

# Fallback to forwarded environment tokens from local build environment
fallback_hf = {escaped_hf_token}
fallback_wandb = {escaped_wandb_key}
if not os.environ.get("HF_TOKEN") and fallback_hf:
    os.environ["HF_TOKEN"] = fallback_hf
    print(">>> [Kaggle Worker] Injected HF_TOKEN from build specification.")
if not os.environ.get("WANDB_API_KEY") and fallback_wandb:
    os.environ["WANDB_API_KEY"] = fallback_wandb
    print(">>> [Kaggle Worker] Injected WANDB_API_KEY from build specification.")

WORKING_DIR = Path("/kaggle/working")
REPO_DIR = WORKING_DIR / "enterprise-llm-finetuning"

# 2. Clone Repository Structure
if not REPO_DIR.exists():
    print(f">>> [Kaggle Worker] Cloning repository from {repo_url} (branch: {git_branch})...")
    subprocess.run(
        ["git", "clone", "--branch", "{git_branch}", "{repo_url}", str(REPO_DIR)],
        check=True
    )
else:
    print(">>> [Kaggle Worker] Repository exists, pulling latest updates...")
    subprocess.run(["git", "pull", "origin", "{git_branch}"], cwd=str(REPO_DIR), check=False)

# 3. Synchronize Active Local Codebase (Unpack embedded src/ package over cloned repo)
src_archive = base64.b64decode({escaped_src}.encode("utf-8"))
with tarfile.open(fileobj=io.BytesIO(src_archive), mode="r:gz") as tar:
    tar.extractall(path=str(REPO_DIR))
print(f">>> [Kaggle Worker] Synchronized active src/ codebase to {{REPO_DIR}}/src")

# Deploy the embedded local experiment configuration directly into cloned repo
REPO_CONFIG = REPO_DIR / "src" / "finetuning_config.yaml"
REPO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
embedded_yaml = {escaped_yaml}
REPO_CONFIG.write_text(embedded_yaml, encoding="utf-8")
print(f">>> [Kaggle Worker] Deployed active experiment configuration to {{REPO_CONFIG}}")

# 4. Install Required Dependencies
print(">>> [Kaggle Worker] Installing fine-tuning dependencies...")
dependencies = [
    "accelerate>=1.14.0",
    "bitsandbytes>=0.49.2",
    "datasets>=3.0.0",
    "peft>=0.14.0",
    "trl>=0.14.0",
    "transformers>=4.48.0",
    "wandb>=0.20.0",
    "structlog>=24.0.0",
    "prometheus-client>=0.20.0",
    "python-dotenv>=0.9.9",
    "pydantic>=2.0.0",
    "pyyaml",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *dependencies], check=True)

# 5. Launch Training & Evaluation Pipeline (Multi-GPU Adaptive)
print(">>> [Kaggle Worker] Launching Fine-Tuning Execution Pipeline...")
env = os.environ.copy()
env["PYTHONPATH"] = str(REPO_DIR)

import torch
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f">>> [Kaggle Worker] Detected {{num_gpus}} available GPU(s).")

if num_gpus > 1:
    print(f">>> [Kaggle Worker] Launching Distributed Data Parallel (DDP) across {{num_gpus}} GPUs via torchrun...")
    launch_cmd = [
        "torchrun",
        f"--nproc_per_node={{num_gpus}}",
        "--master_port=29500",
        "-m", "src.main",
        "--config", str(REPO_CONFIG)
    ]
else:
    print(">>> [Kaggle Worker] Launching single GPU execution via python...")
    launch_cmd = [
        sys.executable,
        "-m", "src.main",
        "--config", str(REPO_CONFIG)
    ]

run_result = subprocess.run(
    launch_cmd,
    cwd=str(REPO_DIR),
    env=env
)

# 6. Sync Outputs to /kaggle/working/ so Kaggle outputs capture artifacts
print(">>> [Kaggle Worker] Exporting artifacts to /kaggle/working/...")
outputs_source = REPO_DIR / "outputs"
if outputs_source.exists():
    for item in outputs_source.iterdir():
        dest = WORKING_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(">>> [Kaggle Worker] Artifacts export complete.")

print(f">>> [Kaggle Worker] Completed with process return code {{run_result.returncode}}.")
if run_result.returncode != 0:
    sys.exit(run_result.returncode)
'''
