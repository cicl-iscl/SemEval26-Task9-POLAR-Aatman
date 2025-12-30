# POLAR @ SemEval-2026

This repository contains my work for **Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization** in **POLAR @ SemEval‑2026**.  
The shared task focuses on detecting and characterizing online polarization across **22 languages** and multiple sociocultural events.  
Official task page: [POLAR @ SemEval‑2026](https://polar-semeval.github.io/).

The current code and experiments primarily target **Subtask 1 – Polarization Detection**, with support data laid out for all three subtasks.

---

## Task Background

Online polarization is the sharp division and hostility between social, political, or identity groups. It is a precursor to hate speech, offensive discourse, and social fragmentation, and can hinder constructive dialogue and social cohesion.  

The POLAR shared task defines three subtasks:
- **Subtask 1 – Polarization Detection**: Decide whether a given text is polarized or not.
- **Subtask 2 – Polarization Type Classification**: Classify the type of polarization (e.g., ideological, identity-based, event-based), depending on the official label set.
- **Subtask 3 – Polarization Manifestation Identification**: Identify how polarization manifests (e.g., us-vs-them framing, derogation, exclusion).

Languages covered (22): Amharic, Arabic, Bengali, Burmese, Chinese, English, German, Hausa, Hindi, Italian, Khmer, Nepali, Odia, Persian, Polish, Punjabi, Russian, Spanish, Swahili, Telugu, Turkish, Urdu.

---

## Repository Structure

- **`data/`**
  - **`dev_phase/`** – Official POLAR data organized by subtask.
    - **`subtask1/`**, **`subtask2/`**, **`subtask3/`**
      - **`train/`**, **`dev/`** – CSV files per language (e.g. `eng.csv`, `arb.csv`, `zho.csv`).
  - Compressed archives (e.g. `dev_phase.zip`) mirroring the released data.

- **`data_eda/`**
  - Exploratory data analysis notebooks, e.g. `subtask1-eda.ipynb`.

- **`experiments/`**
  - **`encoder_models_finetuing/`** – Fine-tuning experiments for multilingual encoders on Subtask 1, e.g.:
    - `polar-subtask1-m-bert.ipynb`
    - `polar-subtask1-xlm-roberta.ipynb`
    - `polar-subtask1-m-deberta.ipynb`
    - `polar-subtask1-m-gte-multilingual-mlm-base.ipynb`
    - `polar-subtask1-glot-500.ipynb`
    - `polar-subtask1-mm-bert.ipynb`
  - **`zero_shot_classification/`** – Zero-shot runs, e.g. `polar-sbutask1-qwen2.5-14b-instruct.ipynb`.
  - **`few_shot_classification/`** – (Placeholder for few-shot experiments; may be populated later.)

- **`submissions/`**
  - Zipped runs formatted for the official Codalab submissions

- **`requirements.in` / `requirements.txt`**
  - Reproducible environment specification.

- **`mid-interim-report.pdf`**
  - Project report with methodological details and intermediate results.

---

## Setup and Environment

### 1. Create and activate a virtual environment
- [install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
```

#### 2. Install dependencies

```bash
uv pip compile requirements.in -o requirements.txt
uv pip install -r requirements.txt
```

This installs Jupyter, pandas, and supporting libraries required to run the notebooks and inspect the data.

#### 3. Launch Jupyter

```bash
jupyter notebook
```

Then open the notebooks under `data_eda/` and `experiments/` to reproduce analysis and experiments.

---

### Data Format

Each CSV file in `data/dev_phase/subtask*/{train,dev}/` corresponds to a **language** and contains at least:
- **`text`** (or equivalent field): the social media/message text.
- **`label`**: the task‑specific label (e.g., polarized vs non‑polarized for Subtask 1).

The exact column names follow the official POLAR data release; see the task description at [POLAR @ SemEval‑2026](https://polar-semeval.github.io/) for full details.

---

### How to Extend This Codebase

- **New models**: Add a new notebook under `experiments/encoder_models_finetuing/` or `experiments/zero_shot_classification/` following the existing structure (data loading, preprocessing, training/evaluation, prediction export).
- **Additional subtasks**: Mirror the Subtask 1 pipeline, but point to `data/dev_phase/subtask2/` or `data/dev_phase/subtask3/` and adjust label handling and metrics.
- **Scripting**: For full automation, you can later refactor notebook logic into Python scripts or modules (e.g. `src/`) and keep notebooks for analysis and visualization only.

---

### Acknowledgements

This work builds on the **POLAR @ SemEval‑2026** shared task and uses data released by the organizers.  
Please refer to the official task page for detailed guidelines, licensing, and any citation instructions: [POLAR @ SemEval‑2026](https://polar-semeval.github.io/).


