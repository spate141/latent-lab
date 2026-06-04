# Local Setup Guide: Ideogram 4 Notebook

Step-by-step instructions for running `notebooks/ideogram4_playground.ipynb`
on this machine (Windows 11 + WSL2, RTX 4090, CUDA 13.0).
This document is updated as new issues are encountered.

---

## System overview

| Component | Details |
|-----------|---------|
| OS | Windows 11 + WSL2 (Ubuntu) |
| GPU | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| CUDA (driver) | 13.0 (KMD 610.47) |
| Python | 3.12.8 (in `.venv`) |
| PyTorch | 2.12.0+cu130 |
| Quantization | nf4 (CUDA → automatically selected) |

---

## Step 1: Clone the repo (if not done)

```bash
git clone https://github.com/ideogram-oss/ideogram4 /mnt/d/ideogram4
cd /mnt/d/ideogram4
```

---

## Step 2: Create the Python venv

From inside `/mnt/d/ideogram4`:

```bash
python3 -m venv .venv
```

This creates `.venv/` in the repo root. Activate it:

```bash
source .venv/bin/activate
```

Your prompt will show `(.venv)`. **All pip commands below assume the venv is active.**

> **WSL2 note:** if `python3` resolves to the system Python (e.g., `/mnt/d/miniforge3/bin/python3`)
> rather than a plain system install, that is fine: the venv gets its own isolated copy.
> Confirm with: `which python` after activation.

---

## Step 3: Install PyTorch with CUDA support

The `pyproject.toml` dependency `torch>=2.11` does not pin a CUDA build, so a plain
`pip install -e .` may pull the CPU-only wheel. Install PyTorch with CUDA first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True  13.0
```

If `torch.cuda.is_available()` returns `False`, see [Troubleshooting › CUDA not found](#cuda-not-found-in-wsl2).

---

## Step 4: Install the ideogram4 package

```bash
pip install -e .
```

This installs all `pyproject.toml` dependencies (`transformers`, `bitsandbytes`,
`safetensors`, `accelerate`, `einops`, `sentencepiece`, `pillow`, `huggingface_hub`,
`requests`) into the venv.

---

## Step 5: Install notebook dependencies

```bash
pip install jupyter ipywidgets ipykernel
```

**Register the venv as a Jupyter kernel** (one-time: this is what makes Jupyter
use the venv's Python where `ideogram4` is installed, rather than the system Python):

```bash
python -m ipykernel install --user --name ideogram4-venv --display-name "Python (ideogram4)"
```

Enable the ipywidgets extension for interactive sliders:

```bash
jupyter nbextension enable --py widgetsnbextension   # classic Notebook
# or
pip install jupyterlab-widgets                        # JupyterLab
```

---

## Step 6: Accept the Hugging Face model gate

The model weights are gated: you must opt in before they can be downloaded.

1. Log into [huggingface.co](https://huggingface.co) in a browser.
2. Open [ideogram-ai/ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4).
3. Click **Agree and access repository** and accept the license.

> Also gate [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8)
> if you plan to run on CPU or MPS.

---

## Step 7: Authenticate HuggingFace

```bash
hf auth login
```

Paste your HF access token when prompted (create one at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **Read** scope).

The token is saved to `~/.cache/huggingface/token` and used automatically on every
subsequent `from_pretrained` call. You only need to do this once.

Verify authentication:

```bash
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

---

## Step 8: Launch Jupyter

From the repo root (with venv active):

```bash
jupyter notebook notebooks/ideogram4_playground.ipynb
# or
jupyter lab
```

Jupyter will print a URL like `http://localhost:8888/?token=...`: paste it into
your **Windows browser**. WSL2 port-forwards to the Windows host automatically.

**Select the right kernel:** in the notebook go to **Kernel → Change kernel →
Python (ideogram4)**. This confirms the venv is in use.

Verify the correct Python is active by running this in any cell:

```python
import sys; print(sys.executable)
# Expected: /mnt/d/ideogram4/.venv/bin/python
```

Then run cells top-to-bottom. If `import ideogram4` fails, the kernel is wrong —
see [Troubleshooting › Jupyter can't find the kernel](#jupyter-cant-find-the-kernel--wrong-python).

---

## Step 9: First model load (weight download)

When you run the **Load pipeline** cell for the first time, `huggingface_hub`
downloads the nf4 weights to `~/.cache/huggingface/hub/models--ideogram-ai--ideogram-4-nf4/`.

Expected download sizes:

| Component | Approx. size |
|-----------|-------------|
| DiT transformer (nf4) | ~5–6 GB |
| Text encoder / Qwen3-VL (nf4) | ~5–6 GB |
| VAE decoder | ~0.5 GB |
| **Total** | **~12–13 GB** |

Download is resumable: if it fails partway, just re-run the cell.

All subsequent runs load from disk; no internet connection needed.

---

## Troubleshooting

### CUDA not found in WSL2

**Symptom:** `torch.cuda.is_available()` returns `False`.

**Cause:** The torch wheel was installed without CUDA support, or the NVIDIA
WSL2 driver isn't set up.

**Fix:**
1. Confirm the Windows NVIDIA driver supports WSL2 CUDA (driver ≥ 515 on Windows side). Check in Windows PowerShell: `nvidia-smi`.
2. Inside WSL2, check that the driver stub is mounted: `ls /usr/lib/wsl/lib/`.
   You should see `libcuda.so`.
3. Reinstall torch with the correct CUDA index URL:
   ```bash
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
4. If your CUDA version differs from 13.0, find the right wheel at
   [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

---

### GatedRepoError / 403 on weight download

**Symptom:**
```
huggingface_hub.errors.GatedRepoError: 403 Client Error ...
Repository ideogram-ai/ideogram-4-nf4 is gated.
```

**Fix:**
1. Visit [ideogram-ai/ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4) and accept the gate.
2. Make sure your HF token has **Read** scope and is authenticated: `hf auth login`.
3. If you exported `HF_TOKEN` manually, confirm it is the same token as the one logged in.

---

### bitsandbytes CUDA mismatch

**Symptom:**
```
RuntimeError: ... bitsandbytes was compiled for CUDA X.Y but PyTorch uses CUDA A.B
```

**Fix:** Install the bitsandbytes wheel that matches your CUDA version:
```bash
pip install bitsandbytes --upgrade
```
If the PyPI wheel doesn't support your CUDA, build from source:
```bash
pip install bitsandbytes --index-url https://huggingface.github.io/bitsandbytes-cuda-versions/
```

---

### ipywidgets not rendering (empty cell output)

**Symptom:** The parameter panel cell produces no visible output.

**Causes and fixes:**

1. **Extension not enabled** (classic Notebook):
   ```bash
   jupyter nbextension enable --py widgetsnbextension
   ```

2. **JupyterLab without the lab extension:**
   ```bash
   pip install jupyterlab-widgets
   jupyter lab build   # only needed for JupyterLab < 3
   ```

3. **Kernel / notebook version mismatch**: restart the kernel and re-run all cells.

4. **VS Code Jupyter extension**: install the `ms-toolsai.jupyter` extension;
   widgets render natively in VS Code's Jupyter UI.

---

### trust_remote_code error

**Symptom:**
```
ValueError: ... requires you to execute the configuration file in that repo.
Set `trust_remote_code=True` if you're fine with that.
```

**Fix:** This is handled internally by `Ideogram4Pipeline.from_pretrained`.
If you see it from a direct `AutoModel.from_pretrained` call elsewhere, add
`trust_remote_code=True`.

---

### Out of VRAM

**Symptom:** `torch.cuda.OutOfMemoryError` during pipeline load or generation.

**Fixes:**
- Close other GPU processes (check with `nvidia-smi`).
- Use a smaller resolution (e.g., 1024×1024 instead of 2048×2048).
- Use `V4_TURBO_12` preset (fewer steps = less peak VRAM).
- On the RTX 4090 (24 GB), the nf4 model should fit comfortably at 1024×1024. At 2048×2048 it may be tight.

---

### Slow first-cell execution (tokenizer warning)

**Symptom:**
```
huggingface_hub.file_download: ... downloading config.json
```
on every run even after weights are cached.

**Cause:** `AutoConfig.from_pretrained` always checks the remote config unless you set `local_files_only=True`. This is a small metadata fetch, not a full weight re-download.

**Fix (optional):** Set `HF_HUB_OFFLINE=1` in your environment to disable all remote checks after the initial download:
```bash
export HF_HUB_OFFLINE=1
```
Add to `~/.bashrc` or your venv activate script to make it permanent.

---

### Jupyter can't find the kernel / wrong Python

**Symptom:** Kernel starts but `import ideogram4` fails with `ModuleNotFoundError`.

**Cause:** The Jupyter kernel is using the system Python rather than the venv.

**Fix:** Register the venv as a kernel:
```bash
# with venv active:
pip install ipykernel
python -m ipykernel install --user --name ideogram4-venv --display-name "Python (ideogram4)"
```
Then select **Python (ideogram4)** from the Jupyter kernel menu.

---

*This document is maintained alongside the notebook. Add new issues at the bottom of the Troubleshooting section as they are encountered.*
