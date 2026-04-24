# v3 Experiment Note (English)

## Setup

- Notebook: `eval_extract_v3.ipynb`
- Data: `query_010.csv` to `query_012.csv`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Methods: `ExpectedAttention (ea)` vs `KVzip (kvzip)`
- Compression ratios: `0.2 / 0.5 / 0.8`
- Result source: `v3/extract_runs.csv`

## Key Results (averaged across queries)

- ratio = 0.2  
  - ea: accuracy = 0.2059, f1_macro = 0.3513  
  - kvzip: accuracy = 0.1241, f1_macro = 0.1686
- ratio = 0.5  
  - ea: accuracy = 0.2597, f1_macro = 0.3307  
  - kvzip: accuracy = 0.0575, f1_macro = 0.0575
- ratio = 0.8  
  - ea: accuracy = 0.3539, f1_macro = 0.2818  
  - kvzip: accuracy = 0.0575, f1_macro = 0.0575

## Observations

- In this v3 run, `ea` consistently outperforms `kvzip` at all three compression ratios.
- `kvzip` drops to near-random performance at `ratio = 0.5` and `0.8` (mean metrics around 0.0575).
- For `ea`, accuracy increases from 0.2 to 0.8, while macro F1 at 0.8 is slightly lower than at 0.2/0.5, suggesting some class-balance instability.
- Sample counts (`n`) are not fully consistent across all `(query, method, ratio)` cells. For a final conclusion, missing samples should be completed first.

## Generated Figures

- `figures/extract_runs_v3_mean_f1_vs_ratio.png`
- `figures/extract_runs_v3_mean_acc_vs_ratio.png`
- `figures/extract_runs_v3_f1_by_query.png`

