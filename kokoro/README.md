# Running `generate_audio.ipynb` with uv

Step-by-step guide to run the Kokoro TTS notebook in a dedicated virtual environment
managed by [`uv`](https://github.com/astral-sh/uv).

---

## Step 0: Clone the Kokoro repository

This notebook must live inside the Kokoro source tree so that the `kokoro` package
can be installed in editable mode from the same directory.

```bash
git clone https://github.com/hexgrad/kokoro.git
cd kokoro
```

Then place `generate_audio.ipynb` in the root of that cloned folder before
continuing with the steps below.

---

## Prerequisites

### 1. Install `uv`

If you don't have `uv` installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify: `uv --version`

### 2. Install `espeak-ng` (system dependency)

`espeak-ng` is used by Kokoro as a phoneme fallback for out-of-dictionary English words.
The pipeline works without it but will log a warning and silently skip uncommon words.

| OS | Command |
|----|---------|
| **Ubuntu / Debian** | `sudo apt-get install espeak-ng` |
| **Fedora / RHEL** | `sudo dnf install espeak-ng` |
| **macOS** | `brew install espeak-ng` |
| **Windows** | Download the `.msi` from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases) (e.g. `espeak-ng-20191129-b702b03-x64.msi`) and run the installer. See the [Windows guide](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md) for advanced configuration. |

---

## Setup

Run these commands once from the **root of this repository**:

```bash
# 1. Create a dedicated virtual environment (Python 3.12 recommended; any 3.10–3.13 works)
uv venv --python 3.12 .venv

# 2. Activate it
#    macOS / Linux:
source .venv/bin/activate
#    Windows (PowerShell):
#    .venv\Scripts\Activate.ps1
#    Windows (cmd):
#    .venv\Scripts\activate.bat

# 3. Install all required packages
uv pip install "kokoro>=0.9.4" soundfile jupyterlab ipykernel ipywidgets numpy

# 4. Register the venv as a Jupyter kernel (so JupyterLab picks it up)
python -m ipykernel install --user --name kokoro-venv --display-name "Python (kokoro-venv)"
```

> **One-time model download**: the first time you run the notebook, Kokoro automatically
> downloads the model weights (~327 MB) from Hugging Face into `~/.cache/huggingface/`.
> Subsequent runs use the local cache.

---

## Run JupyterLab

With the venv active:

```bash
jupyter lab
```

Or without activating explicitly:

```bash
uv run jupyter lab
```

JupyterLab opens in your browser. Open **`generate_audio.ipynb`**, select the
**"Python (kokoro-venv)"** kernel from the top-right kernel picker, then run the cells
top to bottom (**Run → Run All Cells** or `Shift+Enter` cell by cell).

---

## Usage

Edit the **Configuration** cell before running the **Generate Audio** cell:

| Variable | Default | What it controls |
|----------|---------|------------------|
| `MARKDOWN_FILE` | `''` | Path to a `.md` file to narrate (overrides `TEXT` when set) |
| `TEXT` | sample paragraph | The text to synthesise (used when `MARKDOWN_FILE` is empty) |
| `LANG_CODE` | `'a'` | `'a'` = American English, `'b'` = British English |
| `voice_widget` | `af_sky` | Voice preset — select from the dropdown |
| `SPEED` | `1.0` | Playback speed (0.5 – 2.0 typical range) |
| `SPLIT_PATTERN` | `r'\n+'` | Regex used to split text into chunks |
| `OUTPUT_FILENAME` | `'my_audio'` | Base name when narrating `TEXT` (saved as `<voice>_<filename>.wav`) |
| `OUTPUT_DIR` | `'outputs'` | Folder where the output `.wav` is saved |

When `MARKDOWN_FILE` is set, the **Text Source** cell automatically extracts the
`title` from the YAML frontmatter and cleans the body for TTS: code blocks,
HTML/SVG, tables, blockquotes, and links are removed; section headings and prose
are narrated. The output file is named `<voice>_<markdown-filename>.wav`.

### Available voices (English)

| Code | Gender | Accent |
|------|--------|--------|
| `af_heart` | Female | American |
| `af_bella` | Female | American |
| `af_sarah` | Female | American |
| `af_nicole` | Female | American |
| `af_sky` | Female | American |
| `am_michael` | Male | American |
| `am_adam` | Male | American |
| `am_echo` | Male | American |
| `am_eric` | Male | American |
| `am_liam` | Male | American |
| `bf_emma` | Female | British |
| `bf_isabella` | Female | British |
| `bm_george` | Male | British |
| `bm_lewis` | Male | British |

Full list: <https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices>

### Output files

After a successful run you will find:

```
outputs/
  af_sky_my_audio.wav   # <voice>_<OUTPUT_FILENAME>.wav
```

24 kHz mono WAV.

---

## GPU acceleration

### NVIDIA CUDA

CUDA is detected and used automatically when available. No extra configuration needed.

### Apple Silicon (M1/M2/M3/M4)

Enable the PyTorch MPS backend before launching JupyterLab:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter lab
```

Or export it for the whole session:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
jupyter lab
```

---

## Troubleshooting

**`EspeakFallback not Enabled` warning**
: `espeak-ng` is not installed or not on your `PATH`. The notebook still runs; uncommon
  words may be skipped. Install `espeak-ng` (see Prerequisites above) and restart the kernel.

**`ModuleNotFoundError: No module named 'kokoro'`**
: The wrong Python kernel is selected. In JupyterLab, switch to **"Python (kokoro-venv)"**
  via the kernel picker (top-right corner of the notebook).

**`AssertionError` on `LANG_CODE`**
: Only `'a'` and `'b'` are available without extra packages. Other languages require
  `pip install misaki[ja]` or `misaki[zh]` — see the upstream [Kokoro repo](https://github.com/hexgrad/kokoro).

**Python version error**
: Kokoro requires Python `>=3.10, <3.14`. Re-create the venv with `uv venv --python 3.12 .venv`.

**CUDA requested but not available**
: The pipeline auto-selects CPU when CUDA is absent. If you passed `device='cuda'`
  manually and see this error, either install the CUDA-enabled PyTorch build or remove
  the explicit `device` argument.
