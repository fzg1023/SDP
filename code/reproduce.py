# -*- coding: utf-8 -*-
"""SDP 复现入口。

用法（在 release/code 目录下运行）：
  python reproduce.py gear10      # NEU-Gear 10 对 × 5 种子主结果（论文 68.95% ± 0.37%）
  python reproduce.py cwru        # CWRU 三分类（论文 85.96% ± 1.81%）
  python reproduce.py sdust       # SDUST 跨数据集（论文 58.25%）
  python reproduce.py baseline    # 基线对比
  python reproduce.py difficulty  # 难度模型（10 对留一）
  python reproduce.py noise       # 噪声鲁棒性

数据路径可通过环境变量覆盖（默认值为作者本机路径）：
  SDP_GEAR_DATA    NEU-Gear 齿轮数据目录
  SDP_CWRU_DATA    CWRU 轴承数据目录
  SDP_SDUST_DATA   SDUST 轴承数据目录
"""
import os, sys, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "sdp_experiments")

COMMANDS = {
    "gear10": ("train_gear_10pairs_mechproto.py",
               ["--backbone", "small", "--seeds", "5", "--epochs", "50"]),
    "cwru": ("train_fix_cwru.py",
             ["--backbone", "small", "--seeds", "5", "--epochs", "50"]),
    "sdust": ("train_fix_sdust.py",
              ["--backbone", "small", "--epochs", "50"]),
    "baseline": ("baseline_compare.py",
                 ["--backbone", "small"]),
    "difficulty": ("fit_difficulty_model_10pairs.py", []),
    "noise": ("cwru_noise_eval.py", ["--backbone", "small"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=sorted(COMMANDS), help="要复现的实验")
    args, extra = ap.parse_known_args()

    script, base_args = COMMANDS[args.exp]
    script_path = os.path.join(EXP, script)
    if not os.path.exists(script_path):
        print(f"找不到脚本: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, "-u", script_path] + base_args + extra
    print("运行:", " ".join(cmd))
    sys.exit(subprocess.run(cmd, cwd=EXP).returncode)


if __name__ == "__main__":
    main()
