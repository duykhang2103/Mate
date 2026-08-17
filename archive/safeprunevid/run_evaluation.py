"""Reproduce the archived SafePruneVid evaluation on a Modal L4 worker.

Create the credential once with:

    modal secret create huggingface-secret HF_TOKEN=<new-token>

The token is injected as an environment variable and is never embedded in the
image, command line, source tree, or evaluation logs.
"""

from __future__ import annotations

import os
import subprocess

import modal


app = modal.App("prunevid")

app_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .apt_install(
        "python3-dev",
        "build-essential",
        "wget",
        "git",
        "ffmpeg",
        "unzip",
    )
    .pip_install("huggingface_hub", "torch", "torchvision")
    .workdir("/workspace")
    .add_local_file(
        "requirements.clean.txt",
        remote_path="/workspace/requirements.clean.txt",
        copy=True,
    )
    .run_commands("pip install -r requirements.clean.txt")
    .run_commands("pip install -U huggingface_hub")
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".git", "__pycache__"],
        copy=True,
    )
)

output_vol = modal.Volume.from_name("prunevid", create_if_missing=True)
dataset_vol = modal.Volume.from_name(
    "prunevid-dataset", create_if_missing=True
)
# huggingface_secret = modal.Secret.from_name(
#     "huggingface-secret",
#     required_keys=["HF_TOKEN"],
# )


@app.function(
    image=app_image,
    gpu="L4",
    cpu=4.0,
    timeout=86_400,
    volumes={
        "/outputs": output_vol,
        "/dataset": dataset_vol,
    },
    # secrets=[huggingface_secret],
)
def run_pipeline(stage: str = "calibrate"):
    os.chdir("/workspace")
    subprocess.run(
        [
            "mkdir",
            "-p",
            "/dataset/DATAS/Video-MME",
            "/dataset/MODELS",
            "/dataset/Video-MME/results",
            "./DATAS",
        ],
        check=True,
    )
    subprocess.run(
        ["ln", "-snf", "/dataset/MODELS", "./MODELS"],
        check=True,
    )
    subprocess.run(
        [
            "ln",
            "-snf",
            "/dataset/DATAS/Video-MME",
            "./DATAS/Video-MME",
        ],
        check=True,
    )
    subprocess.run(
        [
            "ln",
            "-snf",
            "/dataset/Video-MME/results",
            "./results",
        ],
        check=True,
    )

    evaluation_environment = os.environ.copy()
    evaluation_environment["SAFEPRUNEVID_STAGE"] = stage
    evaluation_environment["SAFEPRUNEVID_ALLOW_ARCHIVED"] = "1"
    process = subprocess.Popen(
        ["bash", "archive/safeprunevid/safeprunevid-eval.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=evaluation_environment,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
    return_code = process.wait()
    output_vol.commit()
    dataset_vol.commit()
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            ["bash", "archive/safeprunevid/safeprunevid-eval.sh"],
        )
    print("Pipeline finished; datasets, models, and results were persisted.")
#     output_vol.commit()
#     # print("✅ Pipeline finished! submission.json safely stored in Volume.")s stored permanently
#     output_vol.commit()
#     # print("✅ Pipeline finished! submission.json safely stored in Volume.")s stored permanently
#     output_vol.commit()
#     # print("✅ Pipeline finished! submission.json safely stored in Volume.")
