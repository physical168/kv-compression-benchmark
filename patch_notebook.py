"""
patch_notebook.py
-----------------
修改 eval_finch_2x2_qwen05b.ipynb，修复 Finch 的两个问题：
1. import 失败可见化
2. FinchPress 需要调用 update_model_and_tokenizer（修复 delimiter token 报错）
3. 添加诊断单元格
运行方式: python patch_notebook.py
"""

import json, copy, pathlib

NB_PATH = pathlib.Path(__file__).parent / "eval_finch_2x2_qwen05b.ipynb"
OUT_PATH = pathlib.Path(__file__).parent / "eval_finch_2x2_qwen05b_fixed.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# ───────────────────────────────────────────────────────────────
# 辅助：构造一个新 cell
# ───────────────────────────────────────────────────────────────
def md_cell(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(lines, cell_id):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": lines,
        "execution_count": None,
        "outputs": [],
        "id": cell_id,
    }

# ───────────────────────────────────────────────────────────────
# 替换 Step 5：Finch adapter
# 关键修复：press.update_model_and_tokenizer(model, tokenizer)
# ───────────────────────────────────────────────────────────────
STEP5_NEW = [
    "# Finch interface via NVIDIA/kvpress FinchPress.\n",
    "# Import errors are now printed loudly so the user notices immediately.\n",
    "\n",
    "# FinchPress window size: number of tokens per FINCH clustering window.\n",
    "# Must be provided; lowered to 16 to handle short reviews correctly.\n",
    "FINCH_WINDOW_SIZE = 16\n",
    "\n",
    "FINCH_AVAILABLE = False\n",
    "FINCH_ERR = \"Finch implementation not available in this runtime\"\n",
    "FinchPress = None\n",
    "\n",
    "try:\n",
    "    from kvpress import FinchPress as _FinchPress\n",
    "    FinchPress = _FinchPress\n",
    "    FINCH_AVAILABLE = True\n",
    "    print(\"[OK] FinchPress imported successfully:\", FinchPress)\n",
    "except Exception as _e:\n",
    "    FINCH_AVAILABLE = False\n",
    "    FINCH_ERR = f\"Finch import failed: {_e}\"\n",
    "    print(\"[WARNING] FinchPress NOT available — Finch-enabled configs will be skipped.\")\n",
    "    print(\"  Detail:\", FINCH_ERR)\n",
    "    print(\"  Fix: !pip install --upgrade kvpress  then restart the runtime.\")\n",
    "\n",
    "\n",
    "def run_generate_with_optional_finch(prompt: str, finch_enabled: bool, compression_ratio: float) -> str:\n",
    "    if finch_enabled and not FINCH_AVAILABLE:\n",
    "        raise RuntimeError(FINCH_ERR)\n",
    "\n",
    "    inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)\n",
    "    if finch_enabled:\n",
    "        input_len = inputs.input_ids.shape[1]\n",
    "        # ── KEY FIX: window_size cannot be larger than input length ──\n",
    "        active_window_size = min(FINCH_WINDOW_SIZE, input_len)\n",
    "        press = FinchPress(compression_ratio=float(compression_ratio))\n",
    "        press.window_size = active_window_size\n",
    "        # update_model_and_tokenizer sets the delimiter token ID (required).\n",
    "        press.update_model_and_tokenizer(model, tokenizer)\n",
    "        with torch.no_grad(), press(model):\n",
    "            out = model.generate(\n",
    "                **inputs,\n",
    "                max_new_tokens=MAX_NEW_TOKENS,\n",
    "                pad_token_id=tokenizer.pad_token_id,\n",
    "            )\n",
    "    else:\n",
    "        with torch.no_grad():\n",
    "            out = model.generate(\n",
    "                **inputs,\n",
    "                max_new_tokens=MAX_NEW_TOKENS,\n",
    "                pad_token_id=tokenizer.pad_token_id,\n",
    "            )\n",
    "    text = tokenizer.decode(out[0], skip_special_tokens=True)\n",
    "    return text\n",
]

# ───────────────────────────────────────────────────────────────
# 新增 Step 5b：Smoke Test（在大循环前快速验证 FinchPress 是否可用）
# ───────────────────────────────────────────────────────────────
STEP5B_MD = [
    "### Step 5b: Smoke test — verify FinchPress works on 1 sample before full run\n",
    "\n",
    "Run this cell **before Step 6**. If it prints `[SMOKE OK]`, Finch is working.\n",
    "If it raises an error, **stop and fix the error first** — do not proceed to Step 6.\n"
]

STEP5B_CODE = [
    "# Quick sanity check: run exactly 1 sample with FinchPress.\n",
    "# If this succeeds without error, the full 4x4 loop will also work.\n",
    "if not FINCH_AVAILABLE:\n",
    "    raise RuntimeError(f'[SMOKE FAIL] FinchPress not available: {FINCH_ERR}')\n",
    "\n",
    "_smoke_prompt = (\n",
    "    'Classify the movie review sentiment. '\n",
    "    'Answer with one word only: positive or negative.\\n\\n'\n",
    "    'Review: This movie was absolutely wonderful.\\n'\n",
    "    'Answer:'\n",
    ")\n",
    "try:\n",
    "    _smoke_out = run_generate_with_optional_finch(_smoke_prompt, finch_enabled=True, compression_ratio=0.5)\n",
    "    print('[SMOKE OK] FinchPress generated output:', repr(_smoke_out[-80:]))\n",
    "    print('Finch is working correctly — safe to run Step 6.')\n",
    "except Exception as _e:\n",
    "    raise RuntimeError(f'[SMOKE FAIL] FinchPress failed on 1 sample: {_e}') from _e\n",
]

# ───────────────────────────────────────────────────────────────
# 新增 Step 6b：诊断单元格
# ───────────────────────────────────────────────────────────────
STEP6B_MD = [
    "### Step 6b: Diagnose — check error counts per config (run after Step 6)"
]

STEP6B_CODE = [
    "# Run this AFTER Step 6 to see how many rows each config got, and what errors occurred.\n",
    "if RUNS_PATH.is_file():\n",
    "    _runs_diag = pd.read_csv(RUNS_PATH)\n",
    "    _runs_diag[\"error\"] = _runs_diag[\"error\"].fillna(\"\").astype(str).str.strip()\n",
    "    _runs_diag[\"has_error\"] = _runs_diag[\"error\"].str.len() > 0\n",
    "\n",
    "    print(\"=== Row counts per config ===\")\n",
    "    print(_runs_diag.groupby([\"config\", \"finch_enabled\", \"use_cpt\", \"has_error\"]).size().to_string())\n",
    "\n",
    "    _finch_errors = _runs_diag[\n",
    "        (_runs_diag[\"finch_enabled\"].astype(str).str.lower().isin([\"true\", \"1\"])) &\n",
    "        (_runs_diag[\"has_error\"])\n",
    "    ][[\"config\", \"error\"]].drop_duplicates()\n",
    "\n",
    "    if len(_finch_errors) == 0:\n",
    "        print(\"\\n[OK] No errors found for Finch-enabled configs.\")\n",
    "    else:\n",
    "        print(\"\\n=== Finch-enabled errors (showing unique messages) ===\")\n",
    "        for _, r in _finch_errors.iterrows():\n",
    "            print(f\"  [{r['config']}] {r['error'][:300]}\")\n",
    "else:\n",
    "    print(\"Runs file not found — run Step 6 first.\")\n",
]

# ───────────────────────────────────────────────────────────────
# 替换 Step 7：汇总时显示被过滤的配置
# ───────────────────────────────────────────────────────────────
STEP7_NEW = [
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import accuracy_score, f1_score\n",
    "\n",
    "runs = pd.read_csv(RUNS_PATH)\n",
    "runs[\"error\"] = runs[\"error\"].fillna(\"\").astype(str).str.strip()\n",
    "ok = runs[(runs[\"error\"] == \"\") | (runs[\"error\"].str.lower() == \"nan\")].copy()\n",
    "ok = ok[ok[\"pred_label\"].notna() & (ok[\"pred_label\"].astype(str).str.len() > 0)]\n",
    "\n",
    "# ── Report any configs that were entirely filtered out ──\n",
    "_all_configs = set(runs[\"config\"].unique())\n",
    "_ok_configs  = set(ok[\"config\"].unique())\n",
    "_dropped = _all_configs - _ok_configs\n",
    "if _dropped:\n",
    "    print(\"[WARNING] The following configs had NO valid predictions and are excluded from summary:\")\n",
    "    for _c in sorted(_dropped):\n",
    "        _sample_err = runs[runs[\"config\"] == _c][\"error\"].dropna().iloc[0] if len(runs[runs[\"config\"] == _c]) else \"(no rows)\"\n",
    "        print(f\"  {_c}: {str(_sample_err)[:300]}\")\n",
    "else:\n",
    "    print(\"[OK] All configs have valid predictions.\")\n",
    "\n",
    "rows = []\n",
    "for cfg in sorted(ok[\"config\"].unique()):\n",
    "    part = ok[ok[\"config\"] == cfg].copy()\n",
    "    if len(part) == 0:\n",
    "        continue\n",
    "    y_true = part[\"gold\"].astype(str).tolist()\n",
    "    y_pred = part[\"pred_label\"].astype(str).tolist()\n",
    "    try:\n",
    "        acc = float(accuracy_score(y_true, y_pred))\n",
    "    except Exception:\n",
    "        acc = float(\"nan\")\n",
    "    try:\n",
    "        f1m = float(f1_score(y_true, y_pred, average=\"macro\"))\n",
    "    except Exception:\n",
    "        f1m = float(\"nan\")\n",
    "    rows.append(\n",
    "        {\n",
    "            \"config\": cfg,\n",
    "            \"finch_enabled\": bool(part[\"finch_enabled\"].iloc[0]),\n",
    "            \"use_cpt\": bool(part[\"use_cpt\"].iloc[0]),\n",
    "            \"accuracy\": acc,\n",
    "            \"f1_macro\": f1m,\n",
    "            \"latency_ms_mean\": float(part[\"latency_ms\"].astype(float).mean()),\n",
    "            \"n\": int(len(part)),\n",
    "        }\n",
    "    )\n",
    "\n",
    "summary = pd.DataFrame(rows).sort_values([\"finch_enabled\", \"use_cpt\"]).reset_index(drop=True)\n",
    "if \"compression_ratio\" in ok.columns and len(summary) > 0:\n",
    "    try:\n",
    "        summary[\"compression_ratio\"] = summary[\"config\"].astype(str).str.extract(r\"_r([0-9.]+)$\")[0].astype(float)\n",
    "    except Exception:\n",
    "        pass\n",
    "summary.to_csv(SUMMARY_PATH, index=False)\n",
    "print(\"Wrote\", SUMMARY_PATH)\n",
    "display(summary)\n",
    "\n",
    "if len(summary) > 0:\n",
    "    fig, ax = plt.subplots(figsize=(8, 4.5))\n",
    "    x = range(len(summary))\n",
    "    ax.bar([i - 0.18 for i in x], summary[\"accuracy\"], width=0.36, label=\"accuracy\")\n",
    "    ax.bar([i + 0.18 for i in x], summary[\"f1_macro\"], width=0.36, label=\"f1_macro\")\n",
    "    ax.set_xticks(list(x))\n",
    "    ax.set_xticklabels(summary[\"config\"], rotation=20, ha=\"right\")\n",
    "    ax.set_ylabel(\"score\")\n",
    "    ax.set_title(\"Finch 4x4 quality\")\n",
    "    ax.legend()\n",
    "    ax.grid(True, axis=\"y\", alpha=0.3)\n",
    "    fig.tight_layout()\n",
    "    fp = FIG_DIR / \"finch_4x4_quality.png\"\n",
    "    fig.savefig(fp, dpi=120)\n",
    "    plt.show()\n",
    "    print(\"Saved\", fp)\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(8, 4.5))\n",
    "    ax.bar(summary[\"config\"], summary[\"latency_ms_mean\"])\n",
    "    ax.set_ylabel(\"mean latency (ms/sample)\")\n",
    "    ax.set_title(\"Finch 4x4 latency\")\n",
    "    ax.grid(True, axis=\"y\", alpha=0.3)\n",
    "    plt.xticks(rotation=20, ha=\"right\")\n",
    "    fig.tight_layout()\n",
    "    fp = FIG_DIR / \"finch_4x4_latency.png\"\n",
    "    fig.savefig(fp, dpi=120)\n",
    "    plt.show()\n",
    "    print(\"Saved\", fp)\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(6, 5))\n",
    "    ax.scatter(summary[\"latency_ms_mean\"], summary[\"f1_macro\"])\n",
    "    for _, r in summary.iterrows():\n",
    "        ax.annotate(r[\"config\"], (r[\"latency_ms_mean\"], r[\"f1_macro\"]), fontsize=8)\n",
    "    ax.set_xlabel(\"mean latency (ms/sample)\")\n",
    "    ax.set_ylabel(\"f1_macro\")\n",
    "    ax.set_title(\"Finch 4x4 trade-off\")\n",
    "    ax.grid(True, alpha=0.3)\n",
    "    fig.tight_layout()\n",
    "    fp = FIG_DIR / \"finch_4x4_tradeoff.png\"\n",
    "    fig.savefig(fp, dpi=120)\n",
    "    plt.show()\n",
    "    print(\"Saved\", fp)\n",
]

# ───────────────────────────────────────────────────────────────
# 执行 patch（基于原始 notebook，不用 fixed 版本）
# ───────────────────────────────────────────────────────────────
cells = nb["cells"]
new_cells = []
step6b_injected = False

step5b_injected = False

for cell in cells:
    src = "".join(cell.get("source", []))

    if cell["cell_type"] == "code" and "FINCH_AVAILABLE = False" in src and "FinchPress = None" in src:
        new_cell = copy.deepcopy(cell)
        new_cell["source"] = STEP5_NEW
        new_cells.append(new_cell)
        print("[PATCH] Replaced Step 5 (Finch adapter + delimiter fix)")
        # Inject Step 5b smoke test immediately after Step 5
        new_cells.append(md_cell(STEP5B_MD))
        new_cells.append(code_cell(STEP5B_CODE, "smoke_5b"))
        step5b_injected = True
        print("[PATCH] Injected Step 5b (smoke test)")

    elif cell["cell_type"] == "code" and "accuracy_score" in src and "f1_score" in src and "summary" in src:
        if not step6b_injected:
            new_cells.append(md_cell(STEP6B_MD))
            new_cells.append(code_cell(STEP6B_CODE, "diag_6b"))
            step6b_injected = True
            print("[PATCH] Injected Step 6b (diagnostic cell)")
        new_cell = copy.deepcopy(cell)
        new_cell["source"] = STEP7_NEW
        new_cells.append(new_cell)
        print("[PATCH] Replaced Step 7 (aggregation + error reporting)")

    else:
        new_cells.append(cell)

nb["cells"] = new_cells

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Patched notebook saved to: {OUT_PATH}")
print("   Upload this file to Colab, delete the old runs CSV, and run again.")
