#!/usr/bin/env python3
"""
从 EgoDex 等 HDF5 读取 camera/intrinsic（3×3 K），换算为 HaMeR 使用的 EXTRA.FOCAL_LENGTH，
并写回 hamer 的两处默认配置（Hydra default.yaml + yacs __init__.py）。

HaMeR 约定（见 demo.py）：
  full 图像上的等效焦距（像素）≈ EXTRA.FOCAL_LENGTH / MODEL.IMAGE_SIZE * max(img_w, img_h)
其中 MODEL.IMAGE_SIZE 默认为 224。为使该值与 HDF5 内参一致，应设：
  EXTRA.FOCAL_LENGTH = mean(fx, fy) * MODEL.IMAGE_SIZE / max(ref_w, ref_h)
ref_w, ref_h 为 K 所对应的标定分辨率（EgoDex RGB 为 1920×1080）。

demo 中精确投影可再加：--camera_fx --camera_fy --camera_cx --camera_cy（本脚本会打印建议值）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# 按需修改：HDF5 路径（服务器示例；本地可改为你的文件）
# -----------------------------------------------------------------------------
HDF5_PATH = "/depot/gsy/hamer/test/add_remove_lid/0.hdf5"

# K 矩阵对应的图像分辨率（EgoDex 官方 RGB）；若你的数据不同请改这里
REF_IMAGE_WIDTH = 1920
REF_IMAGE_HEIGHT = 1080

# HaMeR 默认输入边长（一般勿改，与训练配置一致）
MODEL_IMAGE_SIZE = 224

# 内参在 HDF5 中的数据集路径
INTRINSIC_H5_KEY = "camera/intrinsic"


def _hamer_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_K_from_hdf5(path: Path):
    import numpy as np
    import h5py

    with h5py.File(path, "r") as f:
        if INTRINSIC_H5_KEY not in f:
            raise KeyError(
                f"未找到 {INTRINSIC_H5_KEY!r}，可用键示例: {list(f.keys())[:20]}..."
            )
        K = np.asarray(f[INTRINSIC_H5_KEY][:], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"期望 K 为 (3,3)，得到 {K.shape}")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return K, fx, fy, cx, cy


def compute_hamer_focal_length(fx: float, fy: float, ref_w: int, ref_h: int) -> float:
    f_mean = 0.5 * (fx + fy)
    return f_mean * MODEL_IMAGE_SIZE / max(float(ref_w), float(ref_h))


def patch_yaml_focal(path: Path, focal: float) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"(^\s*FOCAL_LENGTH:\s*)[\d.]+(\s*)$",
        lambda m: f"{m.group(1)}{focal:g}{m.group(2)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(f"无法在 {path} 中替换 FOCAL_LENGTH（匹配次数 {n}）")
    path.write_text(new_text, encoding="utf-8")


def patch_yacs_focal(path: Path, focal: float) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"^_C\.EXTRA\.FOCAL_LENGTH\s*=\s*[\d.]+\s*$",
        f"_C.EXTRA.FOCAL_LENGTH = {focal:g}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(f"无法在 {path} 中替换 _C.EXTRA.FOCAL_LENGTH（匹配次数 {n}）")
    path.write_text(new_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="从 HDF5 提取内参并写入 HaMeR 配置")
    parser.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help="覆盖 HDF5 路径（默认使用脚本内变量 HDF5_PATH）",
    )
    parser.add_argument(
        "--ref-w",
        type=int,
        default=REF_IMAGE_WIDTH,
        help="K 对应的图像宽（像素）",
    )
    parser.add_argument(
        "--ref-h",
        type=int,
        default=REF_IMAGE_HEIGHT,
        help="K 对应的图像高（像素）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印数值，不写配置文件",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="可选：将 fx,fy,cx,cy 与 HaMeR 参数写入该 JSON 路径",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5 or HDF5_PATH)
    if not hdf5_path.is_file():
        print(f"错误: 找不到 HDF5 文件: {hdf5_path}", file=sys.stderr)
        print("请在脚本顶部修改 HDF5_PATH，或使用: --hdf5 /path/to/file.hdf5", file=sys.stderr)
        return 1

    K, fx, fy, cx, cy = load_K_from_hdf5(hdf5_path)
    focal_hamer = compute_hamer_focal_length(fx, fy, args.ref_w, args.ref_h)

    root = _hamer_repo_root()
    yaml_path = root / "hamer" / "configs_hydra" / "experiment" / "default.yaml"
    yacs_path = root / "hamer" / "configs" / "__init__.py"

    print("HDF5:", hdf5_path)
    print("K =\n", K)
    print(f"fx={fx:g} fy={fy:g} cx={cx:g} cy={cy:g}")
    print(f"标定分辨率 ref = {args.ref_w} x {args.ref_h}")
    print(f"将写入 HaMeR EXTRA.FOCAL_LENGTH = {focal_hamer:g}")
    print(
        "验证: full 图默认焦距(像素) ≈ "
        f"{focal_hamer:g} / {MODEL_IMAGE_SIZE} * max(W,H) "
        f"= {(focal_hamer / MODEL_IMAGE_SIZE) * max(args.ref_w, args.ref_h):g}"
    )
    print(
        "\n运行 demo 时若要使用完整 K（推荐与 ViT 检测框同一分辨率）：\n"
        f"  --camera_fx {fx:g} --camera_fy {fy:g} --camera_cx {cx:g} --camera_cy {cy:g}"
    )

    payload = {
        "hdf5_path": str(hdf5_path),
        "intrinsic_K": K.tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "ref_image_width": args.ref_w,
        "ref_image_height": args.ref_h,
        "hamer_extra_focal_length": focal_hamer,
        "model_image_size": MODEL_IMAGE_SIZE,
    }
    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n已写入 JSON: {outp}")

    if args.dry_run:
        print("\n(dry-run，未修改配置文件)")
        return 0

    if not yaml_path.is_file() or not yacs_path.is_file():
        print(f"错误: 找不到 HaMeR 配置: {yaml_path} 或 {yacs_path}", file=sys.stderr)
        return 1

    patch_yaml_focal(yaml_path, focal_hamer)
    patch_yacs_focal(yacs_path, focal_hamer)
    print(f"\n已更新:\n  {yaml_path}\n  {yacs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
