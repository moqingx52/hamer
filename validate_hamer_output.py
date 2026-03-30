from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _bone_lengths(kp3d: np.ndarray) -> np.ndarray:
    # MANO/MediaPipe-like 21 joints tree
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    out = []
    for a, b in edges:
        out.append(np.linalg.norm(kp3d[a] - kp3d[b]))
    return np.asarray(out, dtype=np.float64)


def _draw_points(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], radius: int = 2) -> None:
    for p in pts:
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        cv2.circle(img, (x, y), radius, color, -1)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Validate HaMeR structured JSON with reprojection checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（hamer 仓库根目录，见 Use.md「test/add_remove_lid/0.mp4」）：
  python validate_hamer_output.py --structured_json test/add_remove_lid/hamer_out_0/hamer_structured_results.json --report_json test/add_remove_lid/hamer_out_0/validation_report.json --error_vis_dir test/add_remove_lid/hamer_out_0/validation_vis --max_vis 10
""".strip(),
    )
    p.add_argument("--structured_json", type=str, required=True)
    p.add_argument("--report_json", type=str, required=True)
    p.add_argument("--error_vis_dir", type=str, default=None)
    p.add_argument("--max_vis", type=int, default=5)
    p.add_argument("--topk_bad", type=int, default=20, help="Top-K worst samples for visualization candidates")
    args = p.parse_args()

    with open(args.structured_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("structured_json should be a list of hand records")

    n_total = len(data)
    nan_or_inf = 0
    bad_joint_count = 0
    z_non_positive = 0
    reproj_errors: list[float] = []
    per_sample_reproj: list[tuple[float, dict[str, Any]]] = []
    betas_by_side: dict[int, list[np.ndarray]] = {0: [], 1: []}
    bone_lengths_by_side: dict[int, list[np.ndarray]] = {0: [], 1: []}
    side_by_frame_person: dict[tuple[int, int], list[int]] = {}

    for rec in data:
        frame_idx = int(rec.get("frame_idx", -1))
        person_id = int(rec.get("person_id", -1))
        is_right = int(rec.get("is_right", 0))
        key = (frame_idx, person_id)
        side_by_frame_person.setdefault(key, []).append(is_right)

        kp3d = np.asarray(rec.get("keypoints_3d_cam", []), dtype=np.float64)
        if kp3d.shape != (21, 3):
            bad_joint_count += 1
        else:
            if np.any(~np.isfinite(kp3d)):
                nan_or_inf += 1
            if np.any(kp3d[:, 2] <= 0):
                z_non_positive += 1
            bone_lengths_by_side.setdefault(is_right, []).append(_bone_lengths(kp3d))

        betas = np.asarray(rec.get("betas", []), dtype=np.float64).reshape(-1)
        if betas.size > 0 and np.all(np.isfinite(betas)):
            betas_by_side.setdefault(is_right, []).append(betas)

        kp2d = np.asarray(rec.get("hand_keypoints_2d", []), dtype=np.float64)
        if kp2d.shape[0] == 21 and kp3d.shape == (21, 3):
            fx_fy = rec.get("focal_length_px", [None, None])
            cc = rec.get("camera_center_px", [None, None])
            fx = _safe_float(fx_fy[0]) if len(fx_fy) > 0 else float("nan")
            fy = _safe_float(fx_fy[1]) if len(fx_fy) > 1 else float("nan")
            cx = _safe_float(cc[0]) if len(cc) > 0 else float("nan")
            cy = _safe_float(cc[1]) if len(cc) > 1 else float("nan")
            if np.isfinite([fx, fy, cx, cy]).all():
                Z = np.clip(kp3d[:, 2], 1e-8, None)
                reproj = np.stack([fx * kp3d[:, 0] / Z + cx, fy * kp3d[:, 1] / Z + cy], axis=-1)
                vis = kp2d[:, 2] > 0.1 if kp2d.shape[1] > 2 else np.ones((21,), dtype=bool)
                if np.any(vis):
                    err = np.linalg.norm(reproj[vis] - kp2d[vis, :2], axis=1)
                    sample_err = float(np.mean(err))
                    reproj_errors.extend(err.tolist())
                    per_sample_reproj.append((sample_err, rec))

    betas_stats = {}
    for side, arr in betas_by_side.items():
        if len(arr) <= 1:
            betas_stats[str(side)] = {"std_mean": None}
            continue
        mats = np.stack(arr, axis=0)
        betas_stats[str(side)] = {"std_mean": float(np.mean(np.std(mats, axis=0)))}

    bone_cv_stats = {}
    for side, arr in bone_lengths_by_side.items():
        if len(arr) <= 1:
            bone_cv_stats[str(side)] = {"cv_mean": None}
            continue
        mats = np.stack(arr, axis=0)
        mean_len = np.mean(mats, axis=0)
        std_len = np.std(mats, axis=0)
        cv = std_len / (mean_len + 1e-8)
        bone_cv_stats[str(side)] = {"cv_mean": float(np.mean(cv))}

    side_flip_count = 0
    for _, sides in side_by_frame_person.items():
        if len(set(sides)) > 1:
            side_flip_count += 1

    reproj_summary = None
    if len(reproj_errors) > 0:
        e = np.asarray(reproj_errors, dtype=np.float64)
        reproj_summary = {
            "count": int(e.size),
            "mean_px": float(np.mean(e)),
            "median_px": float(np.median(e)),
            "p95_px": float(np.percentile(e, 95)),
            "quality_hint": (
                "good(<10px)" if np.percentile(e, 95) < 10
                else "usable(10-25px)" if np.percentile(e, 95) <= 25
                else "risky(>25px)"
            ),
        }

    must_pass = {
        "no_nan_inf": nan_or_inf == 0,
        "all_have_21_joints": bad_joint_count == 0,
        "all_z_positive": z_non_positive == 0,
        "betas_stable": all(
            (v.get("std_mean") is None) or (v.get("std_mean") < 1e-3)
            for v in betas_stats.values()
        ),
        "bone_cv_small": all(
            (v.get("cv_mean") is None) or (v.get("cv_mean") < 0.1)
            for v in bone_cv_stats.values()
        ),
        "side_not_flip_same_frame_person": side_flip_count == 0,
    }

    report = {
        "input_file": str(Path(args.structured_json).resolve()),
        "num_samples": n_total,
        "must_pass": must_pass,
        "summary_counts": {
            "nan_or_inf_samples": nan_or_inf,
            "bad_joint_count_samples": bad_joint_count,
            "z_non_positive_samples": z_non_positive,
            "side_flip_conflicts": side_flip_count,
        },
        "reprojection": reproj_summary,
        "betas_stats": betas_stats,
        "bone_length_cv": bone_cv_stats,
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.error_vis_dir:
        vis_dir = Path(args.error_vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(per_sample_reproj, key=lambda x: x[0], reverse=True)[: max(1, args.topk_bad)]
        dumped = 0
        for mean_err, rec in candidates:
            if dumped >= args.max_vis:
                break
            img_path = rec.get("image_path")
            if not img_path:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            kp2d = np.asarray(rec.get("hand_keypoints_2d", []), dtype=np.float64)
            kp3d = np.asarray(rec.get("keypoints_3d_cam", []), dtype=np.float64)
            if kp2d.shape[0] != 21 or kp3d.shape != (21, 3):
                continue
            fx, fy = rec.get("focal_length_px", [None, None])
            cx, cy = rec.get("camera_center_px", [None, None])
            if None in [fx, fy, cx, cy]:
                continue
            Z = np.clip(kp3d[:, 2], 1e-8, None)
            reproj = np.stack([float(fx) * kp3d[:, 0] / Z + float(cx), float(fy) * kp3d[:, 1] / Z + float(cy)], axis=-1)
            _draw_points(img, kp2d[:, :2], (0, 255, 0), 2)      # GT/ViTPose
            _draw_points(img, reproj, (0, 0, 255), 2)           # Reprojected
            txt = f"f={rec.get('frame_idx')} p={rec.get('person_id')} err={mean_err:.2f}px"
            cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            out_img = vis_dir / f"err_{dumped:02d}_f{rec.get('frame_idx')}_p{rec.get('person_id')}.jpg"
            cv2.imwrite(str(out_img), img)
            dumped += 1

    print(f"Validation report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

