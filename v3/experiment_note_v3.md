# v3 实验简要记录

## 实验设置

- Notebook: `eval_extract_v3.ipynb`
- 数据: `query_010.csv`–`query_012.csv`
- 模型: `Qwen/Qwen2.5-0.5B-Instruct`
- 方法: `ExpectedAttention (ea)` vs `KVzip (kvzip)`
- 压缩率: `0.2 / 0.5 / 0.8`
- 结果源文件: `v3/extract_runs.csv`

## 关键结果（按 query 平均）

- ratio=0.2  
  - ea: accuracy=0.2059, f1_macro=0.3513  
  - kvzip: accuracy=0.1241, f1_macro=0.1686
- ratio=0.5  
  - ea: accuracy=0.2597, f1_macro=0.3307  
  - kvzip: accuracy=0.0575, f1_macro=0.0575
- ratio=0.8  
  - ea: accuracy=0.3539, f1_macro=0.2818  
  - kvzip: accuracy=0.0575, f1_macro=0.0575

## 观察

- 在本次 v3 结果中，`ea` 在三个压缩率下均显著高于 `kvzip`。
- `kvzip` 在 `ratio=0.5/0.8` 上几乎退化到接近随机（均值仅约 0.0575）。
- `ea` 的 accuracy 随 ratio 从 0.2 -> 0.8 上升，但 macro F1 在 0.8 略低于 0.2/0.5，说明类别平衡表现仍有波动。
- 当前 `n` 在不同 query/方法下不完全一致（部分组合样本数偏小），后续建议先补齐缺失样本再做最终结论。

## 产出图

- `figures/extract_runs_v3_mean_f1_vs_ratio.png`
- `figures/extract_runs_v3_mean_acc_vs_ratio.png`
- `figures/extract_runs_v3_f1_by_query.png`

