# Sycophantic AI makes human interaction feel more effortful and less satisfying over time

This repository contains the code and data for the five studies reported in the submission.

**Paper:** https://arxiv.org/pdf/2605.07912

## Structure

- `study_01_support_importance/` — analysis script and data for Study 1
- `study_02_perceptions_sycophantic/` — analysis script and data for Study 2
- `study_03_anticipation_close_others/` — analysis script and data for Study 3
- `study_04_longitudinal/` — longitudinal study with multiple analysis scripts (`analysis/`) and data files (`data/`)
- `study_05_choice_preferences/` — analysis script and data for Study 5
- `supp_info.pdf` — supplementary information

Each study folder contains the analysis code (`analyze_study_*.py`) and a `data/` directory with the corresponding CSV(s).

## Setup

Clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Each analysis script expects to be run from inside its study folder, e.g.:

```bash
cd study_01_support_importance
python analyze_study_01.py
```

Study 4's analyses live under `study_04_longitudinal/analysis/` and read from `study_04_longitudinal/data/`; run them from `study_04_longitudinal/analysis/` (e.g. `cd study_04_longitudinal/analysis && python longitudinal_results.py`).
