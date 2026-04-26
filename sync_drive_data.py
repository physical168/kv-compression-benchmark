import os
import shutil
import pathlib

# ───────────────────────────────────────────────────────────────
# 配置：修改为你实际的 Drive 目录名
# ───────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
DRIVE_SUBDIR = "kv-compression-benchmark/finch_4x4_qwen05b"
FILE_NAME = "finch_4x4_runs.csv"
TARGET_PATH = PROJECT_ROOT / FILE_NAME

def sync_from_local_drive():
    """
    尝试从 Windows 上挂载的 Google Drive 虚拟盘直接复制文件。
    默认挂载字母通常是 G:
    """
    # 尝试常见的挂载盘符
    for drive_letter in ["G", "H", "I", "F"]:
        drive_path = pathlib.Path(f"{drive_letter}:/My Drive") / DRIVE_SUBDIR / FILE_NAME
        if drive_path.exists():
            print(f"找到同步盘路径: {drive_path}")
            shutil.copy2(drive_path, TARGET_PATH)
            print(f"✅ 文件已成功同步到本地: {TARGET_PATH}")
            return True
    return False

def instructions_for_manual_download():
    print("\n[INFO] 未检测到本地 Google Drive 同步盘 (G:\\)。")
    print(f"请手动从 Colab/Google Drive 下载: {DRIVE_SUBDIR}/{FILE_NAME}")
    print(f"并将其放在此项目根目录下: {PROJECT_ROOT}")

if __name__ == "__main__":
    print("正在尝试获取 Drive 运行结果数据...")
    if not sync_from_local_drive():
        instructions_for_manual_download()
    else:
        # 如果获取成功，顺便打印个简单的统计信息
        import pandas as pd
        df = pd.read_csv(TARGET_PATH)
        print("\n=== 本地数据概览 ===")
        print(f"总行数: {len(df)}")
        print("各配置行数统计:")
        print(df.groupby('config').size())
