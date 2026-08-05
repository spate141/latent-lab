# Running `shieldstral_policy_lab.ipynb` locally

This guide sets up the Shieldstral Policy Lab with [`uv`](https://docs.astral.sh/uv/),
Python 3.13, and an NVIDIA GPU. The notebook runs
[`mistralai/Shieldstral-1.0-3B`](https://huggingface.co/mistralai/Shieldstral-1.0-3B)
locally through Transformers—no inference API or separate model server is used.

## What the notebook demonstrates

- Interactive policy evaluation for text, prompt/response pairs, and image+text inputs, exposed
  as three independent, always-visible example widgets (Policy wording, Refusal detection, Image
  policy) plus a separate preset-driven playground for building your own cases
- Continuous policy-match probabilities derived from the model's `yes`/`no` logits
- Side-by-side evaluation of the same documents against multiple natural-language policies
- Decision-threshold exploration without rerunning inference
- Local latency, VRAM, and RAM measurements

## Tested target

| Component | Target |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu) or native Linux |
| Python | 3.13 |
| GPU | NVIDIA GPU with at least 16 GB VRAM |
| Reference GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| PyTorch build | CUDA 13.0 |
| Model dtype | BF16 |

Shieldstral officially fits on a single 16 GB NVIDIA GPU in BF16. CPU fallback exists in
the notebook, but it is expected to be much slower.

## 1. Clone the repository

```bash
git clone https://github.com/spate141/latent-lab.git
cd latent-lab/shieldstral
```

## 2. Install `uv`

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell if `uv` is not immediately on `PATH`, then verify it:

```bash
uv --version
```

## 3. Create a Python 3.13 environment

Run these commands from the `shieldstral` directory:

```bash
uv python install 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate
```

Confirm the interpreter:

```bash
python --version
# Expected: Python 3.13.x
```

## 4. Install PyTorch with CUDA 13.0

Install the CUDA-enabled PyTorch wheel before the remaining dependencies:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verify that PyTorch sees the GPU:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output includes `CUDA: True` and your NVIDIA GPU name. The CUDA version reported by
`nvidia-smi` is the maximum supported by the driver; it does not have to exactly equal the PyTorch
wheel's bundled CUDA runtime.

If CUDA 13.0 is not appropriate for your driver or platform, select the current matching command at
<https://pytorch.org/get-started/locally/> instead.

> **WSL2 note:** use an NVIDIA Windows driver with WSL support. Do not install a second Linux NVIDIA
> display driver inside WSL.

## 5. Install notebook dependencies

```bash
uv pip install --upgrade \
  "transformers[torch,mistral-common]" \
  "mistral-common>=1.11.5" \
  accelerate \
  huggingface_hub \
  pillow \
  ipywidgets \
  jupyterlab \
  ipykernel \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  psutil
```

Shieldstral is a new model architecture, so use the current Transformers release instead of an older
environment-pinned version.

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user \
  --name shieldstral-venv \
  --display-name "Python (shieldstral)"
```

## 6. Launch JupyterLab

With the environment active:

```bash
jupyter lab shieldstral_policy_lab.ipynb
```

Or launch without activating the environment explicitly:

```bash
uv run jupyter lab shieldstral_policy_lab.ipynb
```

Select **Python (shieldstral)** in the kernel picker, then run the notebook from top to bottom.
The first model-load run downloads the weights into the Hugging Face cache. Later sessions reuse
the cached files.

## Recommended run order

1. Confirm CUDA, GPU, and BF16 in the hardware cell.
2. Load the tokenizer and model once.
3. Run the **preset playground** cell, then run the **three independent example widgets** cell—
   each of the three (Policy wording, Refusal detection, Image policy) renders pre-filled and
   ready to run on its own, with its own inputs, threshold slider, and output panel.
4. Click **Analyze locally** on the **Policy wording** widget.
5. Click **Analyze locally** on the **Refusal detection** widget and notice that a high `yes`
   score means the refusal policy matched—it does not mean the response was unsafe.
6. Upload a local image in the **Image policy** widget, then click **Analyze locally**.
7. Run the policy matrix.
8. Move the threshold slider without rerunning inference.
9. Refresh the performance summary.

## Troubleshooting

### `ImportError: cannot import name Mistral3ForConditionalGeneration`

The Transformers installation is too old or the notebook is using the wrong kernel:

```bash
source .venv/bin/activate
uv pip install --upgrade "transformers[torch,mistral-common]" "mistral-common>=1.11.5"
python -c "from transformers import Mistral3ForConditionalGeneration; print('import ok')"
```

Restart Jupyter and select **Python (shieldstral)**.

### `torch.cuda.is_available()` is `False`

1. Run `nvidia-smi` in Windows and inside WSL.
2. Confirm that WSL GPU support is enabled in the Windows NVIDIA driver.
3. Confirm that the environment contains a CUDA PyTorch build:

   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```

4. Reinstall the matching PyTorch wheel if `torch.version.cuda` prints `None`.

### CUDA out of memory

- Close other GPU-heavy applications and model servers.
- Restart the notebook kernel to release stale allocations.
- Confirm that only one copy of the notebook kernel is running.
- Avoid loading another large model in the same kernel.

### Image upload does not appear

Make sure `ipywidgets` is installed in the selected kernel:

```bash
uv pip install --upgrade ipywidgets jupyterlab
```

Restart JupyterLab after installation.

### The policy matrix is slow on its first run

The first inference can include CUDA kernel initialization and is usually slower than subsequent
runs. The matrix intentionally evaluates every document-policy pair separately because Shieldstral
expects one yes/no policy per query.

## Model behavior and responsible use

- `policy_match_probability` is conditional on the exact instruction, query, and document.
- Use one yes/no policy per query.
- Calibrate production thresholds on representative, human-reviewed data.
- Do not use the notebook as the sole decision-maker for consequential moderation actions.
- Review the model card's limitations before deployment, especially for multilingual, obfuscated,
  adversarial, or very long inputs.

## References

- [Shieldstral announcement](https://mistral.ai/news/shieldstral/)
- [Shieldstral model card and usage examples](https://huggingface.co/mistralai/Shieldstral-1.0-3B)
- [Shieldstral technical report](https://arxiv.org/abs/2607.25857)

