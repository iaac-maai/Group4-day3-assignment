---
license: mit
library_name: sklearn
tags:
  - sklearn
  - clustering
  - unsupervised-learning
  - kmeans
  - energy
  - built-environment
  - iaac-maai
  - ai-methods-1
datasets:
  - iaac-maai/day3-group4-household-energy-trajectories
---

# Household energy typologies: grouping homes by their gas and electricity trajectories

This model assigns a gas-heated home in England or Wales to one of **4 energy-behaviour groups**. It uses how much gas and electricity the home uses today and how that use changed between 2005 and 2024. It is a scikit-learn pipeline (`StandardScaler → KMeans`, k = 4) fitted on 29,390 homes from the UK NEED anonymised dataset. We built it as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full analysis):** https://github.com/iaac-maai/Group4-day3-assignment
- **Team:** Group 4 – Luis, Fawad, Hetashri, Daria

## Intended use
- **Intended users and use:** energy-efficiency programmes and local authorities that want to **segment a housing stock for outreach**. For example: which homes to offer insulation first, and which to check for fuel poverty.
- **Out-of-scope uses:** homes not heated by gas; homes outside England and Wales; periods outside 2005–2024; any judgement about an individual household (the groups overlap strongly); predicting future consumption; any decision that affects a household's rights or tariffs.

## Model details
| | |
|---|---|
| **Task** | Clustering (unsupervised) |
| **Algorithm** | `KMeans` |
| **Key hyperparameters** | `n_clusters=4`, `n_init=10`, `random_state=42` |
| **Preprocessing** | log transforms done **before** the model (see Input features); `StandardScaler` inside the pipeline |
| **Assigner** | not needed (`KMeans.predict`) |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` |
| **Library versions** | scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, Python 3.14 |

## Input features
A `pandas.DataFrame` with exactly these columns. Average each fuel over the valid annual readings (at least 2) in each window: early = 2005–08, pre-crisis = 2019–21, recent = 2022–24.

| Column | Type | Unit | Formula |
|---|---|---|---|
| `log_gas_recent` | float | log10 kWh/yr | log10(gas recent) |
| `log_elec_recent` | float | log10 kWh/yr | log10(electricity recent) |
| `gas_change_long` | float | log2 ratio | log2(gas pre-crisis / gas early) |
| `gas_change_crisis` | float | log2 ratio | log2(gas recent / gas pre-crisis) |
| `elec_change_long` | float | log2 ratio | log2(electricity recent / electricity early) |

## Output and clusters
`.predict()` returns an integer from 0 to 3:

| Cluster | Name | Share | Defining features (medians) |
|---|---|---|---|
| 0 | high users, rising electricity | 30 % | gas 12,900 kWh; electricity 4,400 kWh, +24 % since 2005 |
| 1 | low users, deep crisis cuts | 17 % | gas 5,470 kWh; gas −40 % in 2022; most deprived areas |
| 2 | early reducers, now low gas | 17 % | gas −53 % from 2005–08 to 2019–21, −2 % in the crisis; most insulated |
| 3 | high gas, slow reducers | 36 % | gas 12,700 kWh; −16 % long term, −13 % in the crisis; fewest EPC A–C |

## Training data
NEED anonymised 50k sample (UK DESNZ, Open Government Licence v3.0), as cleaned by Group 4 on Days 1–2. Published with clusters as [iaac-maai/day3-group4-household-energy-trajectories](https://huggingface.co/datasets/iaac-maai/day3-group4-household-energy-trajectories). One row = one home. We kept the gas-heated homes with valid (`V`) readings in all three windows: 29,390 of 40,445.

## Method and evaluation
- We compared PCA, t-SNE and UMAP for structure, and K-Means (k = 2…10), Ward hierarchical clustering and DBSCAN for grouping.
- **No natural clusters:** silhouette ≤ 0.22 for every k, the dendrogram has no dominant merges, and DBSCAN finds one dense cluster plus 7 % noise.
- **k = 4** was chosen for interpretability and the drop in Davies–Bouldin (1.62 → 1.50). It gives a silhouette of 0.18, and the smallest group is 17 %.
- **Stability:** ARI 0.98 across 10 seeds and ≥ 0.94 across 10 subsamples of 80 %. Agreement with Ward (k = 4): ARI 0.27.

## Limitations and biases
- The groups are a **reproducible cut of a continuum**. Many homes sit near a border.
- Annual readings cannot tell efficiency gains apart from changes in occupancy. "Deep crisis cuts" may mean under-heating; this is a hypothesis, not a measurement.
- Homes connected after 2008 or without enough valid readings are excluded, which biases the sample towards older, stable households.
- NEED rounds readings, and flagged readings were removed.

## How to use
```python
import joblib, numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

model = joblib.load(hf_hub_download("iaac-maai/day3-group4-household-energy-typologies", "model.joblib"))
NAMES = {0: "high users, rising electricity", 1: "low users, deep crisis cuts",
         2: "early reducers, now low gas", 3: "high gas, slow reducers"}

gas_early, gas_pre, gas_recent = 18000, 16500, 14000      # kWh/yr averages
elec_early, elec_recent = 3500, 2300
home = pd.DataFrame([{
    "log_gas_recent": np.log10(gas_recent), "log_elec_recent": np.log10(elec_recent),
    "gas_change_long": np.log2(gas_pre / gas_early), "gas_change_crisis": np.log2(gas_recent / gas_pre),
    "elec_change_long": np.log2(elec_recent / elec_early)}])
print(NAMES[model.predict(home)[0]])
```

## Reproducibility
Notebook `notebook/unsupervised_learning.ipynb` and `requirements.txt` in the GitHub repository.

## Security note
`.joblib` files are pickles and can run code when loaded. Only load models from sources you trust.

## Authors, course and licence
Group 4, AI Methods I, IAAC MaAI 2026/27. Model: MIT. Data: Open Government Licence v3.0 (UK DESNZ, NEED).
