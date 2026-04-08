from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional, Tuple
import re
from collections import deque


class TemporalSmoother:
    """
    Exponential Moving Average (EMA) smoother for hand pose sequences.
    Smooths position, rotation, and MANO parameters across frames.
    """
    
    def __init__(self, alpha=0.3, min_score=0.5):
        """
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother but more lag.
            min_score: Minimum detection score to update the smoother.
        """
        self.alpha = alpha
        self.min_score = min_score
        self.prev_p_wrist = None
        self.prev_R_wrist = None
        self.prev_cam_t = None
        self.prev_global_orient = None
        self.prev_hand_pose = None
        self.prev_betas = None
        self.initialized = False
    
    def smooth_rotation_matrix(self, R_new, R_prev):
        """
        Smooth rotation matrices using quaternion slerp approximation.
        Convert to axis-angle, interpolate, convert back.
        """
        from scipy.spatial.transform import Rotation as R_scipy
        
        # Convert rotation matrices to quaternions
        quat_new = R_scipy.from_matrix(R_new).as_quat()
        quat_prev = R_scipy.from_matrix(R_prev).as_quat()
        
        # Ensure shortest path (quaternion sign ambiguity)
        if np.dot(quat_new, quat_prev) < 0:
            quat_new = -quat_new
        
        # Slerp interpolation
        t = self.alpha
        # Simple lerp for small angles (good enough for consecutive frames)
        quat_smooth = (1 - t) * quat_prev + t * quat_new
        quat_smooth = quat_smooth / (np.linalg.norm(quat_smooth) + 1e-8)
        
        # Convert back to rotation matrix
        R_smooth = R_scipy.from_quat(quat_smooth).as_matrix()
        return R_smooth.astype(np.float64)
    
    def smooth_vector(self, v_new, v_prev):
        """Simple EMA for vectors."""
        return self.alpha * v_new + (1 - self.alpha) * v_prev
    
    def update(self, p_wrist, R_wrist, cam_t, global_orient, hand_pose, betas, score=None):
        """
        Update smoother with new frame data.
        
        Args:
            p_wrist: wrist position (3,)
            R_wrist: wrist rotation (3,3)
            cam_t: camera translation (3,)
            global_orient: MANO global orientation (1,3,3) or (3,3)
            hand_pose: MANO hand pose parameters
            betas: MANO shape parameters
            score: detection confidence (optional)
            
        Returns:
            Smoothed values (same format as inputs)
        """
        # Skip smoothing if score is too low
        if score is not None and score < self.min_score:
            return p_wrist, R_wrist, cam_t, global_orient, hand_pose, betas
        
        if not self.initialized:
            # First frame: initialize
            self.prev_p_wrist = p_wrist.copy()
            self.prev_R_wrist = R_wrist.copy()
            self.prev_cam_t = cam_t.copy()
            self.prev_global_orient = global_orient.copy()
            self.prev_hand_pose = hand_pose.copy()
            self.prev_betas = betas.copy()
            self.initialized = True
            return p_wrist, R_wrist, cam_t, global_orient, hand_pose, betas
        
        # Smooth position
        p_wrist_smooth = self.smooth_vector(p_wrist, self.prev_p_wrist)
        
        # Smooth camera translation
        cam_t_smooth = self.smooth_vector(cam_t, self.prev_cam_t)
        
        # Smooth rotation (wrist)
        R_wrist_smooth = self.smooth_rotation_matrix(R_wrist, self.prev_R_wrist)
        
        # Smooth global orientation (rotation matrix)
        go_shape = global_orient.shape
        global_orient_reshaped = global_orient.reshape(-1, 3, 3)
        prev_go_reshaped = self.prev_global_orient.reshape(-1, 3, 3)
        global_orient_smooth_list = []
        for i in range(global_orient_reshaped.shape[0]):
            R_smooth = self.smooth_rotation_matrix(
                global_orient_reshaped[i], prev_go_reshaped[i]
            )
            global_orient_smooth_list.append(R_smooth)
        global_orient_smooth = np.stack(global_orient_smooth_list).reshape(go_shape)
        
        # Smooth hand pose (assuming it's already in rotation matrix format)
        hp_shape = hand_pose.shape
        if len(hp_shape) >= 2 and hp_shape[-2:] == (3, 3):
            # It's rotation matrices
            hand_pose_reshaped = hand_pose.reshape(-1, 3, 3)
            prev_hp_reshaped = self.prev_hand_pose.reshape(-1, 3, 3)
            hand_pose_smooth_list = []
            for i in range(hand_pose_reshaped.shape[0]):
                R_smooth = self.smooth_rotation_matrix(
                    hand_pose_reshaped[i], prev_hp_reshaped[i]
                )
                hand_pose_smooth_list.append(R_smooth)
            hand_pose_smooth = np.stack(hand_pose_smooth_list).reshape(hp_shape)
        else:
            # It's a vector (e.g., axis-angle), use simple EMA
            hand_pose_smooth = self.smooth_vector(hand_pose, self.prev_hand_pose)
        
        # Smooth betas (simple EMA)
        betas_smooth = self.smooth_vector(betas, self.prev_betas)
        
        # Update state
        self.prev_p_wrist = p_wrist_smooth.copy()
        self.prev_R_wrist = R_wrist_smooth.copy()
        self.prev_cam_t = cam_t_smooth.copy()
        self.prev_global_orient = global_orient_smooth.copy()
        self.prev_hand_pose = hand_pose_smooth.copy()
        self.prev_betas = betas_smooth.copy()
        
        return p_wrist_smooth, R_wrist_smooth, cam_t_smooth, global_orient_smooth, hand_pose_smooth, betas_smooth
    
    def reset(self):
        """Reset smoother state (e.g., when switching subjects)."""
        self.initialized = False
        self.prev_p_wrist = None
        self.prev_R_wrist = None
        self.prev_cam_t = None
        self.prev_global_orient = None
        self.prev_hand_pose = None
        self.prev_betas = None


def _wrist_pose_cam_to_base(
    p_cam: np.ndarray, R_cam: np.ndarray, T_cam2base: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """SE(3): p_b = R_cb p_c + t_cb, R_b = R_cb R_c (wrist pose in camera frame -> robot base)."""
    R_cb = T_cam2base[:3, :3]
    t_cb = T_cam2base[:3, 3]
    p_base = R_cb @ p_cam + t_cb
    R_base = R_cb @ R_cam
    return p_base, R_base


def _parse_intrinsics_file(path: str) -> Dict[str, float]:
    """
    Parse camera intrinsics from a text file.

    Supported examples:
      Camera1.fx: 511.883062
      Camera1.fy: 511.790371
      Camera1.cx: 1081.052472
      Camera1.cy: 1081.141909

    Also supports lines like:
      fx=511.88
      fy: 511.79
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Intrinsics file not found: {p}')

    text = p.read_text(encoding='utf-8', errors='ignore')
    # Match optionally prefixed "CameraX." then one of fx/fy/cx/cy then ":" or "=" then a float.
    pattern = re.compile(
        r'(?im)^\s*(?:[A-Za-z0-9_]+\.)?\s*(fx|fy|cx|cy)\s*[:=]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$'
    )
    values: Dict[str, float] = {}
    for m in pattern.finditer(text):
        k = m.group(1).lower()
        v = float(m.group(2))
        values[k] = v

    missing = [k for k in ('fx', 'fy', 'cx', 'cy') if k not in values]
    if missing:
        raise ValueError(
            f'Intrinsics file is missing keys: {missing}. '
            f'Expected lines like "Camera1.fx: 511.883062". File: {p}'
        )
    return values


def _write_camera_json(path: str, fx: float, fy: float, cx: float, cy: float) -> None:
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(cx),
        'cy': float(cy),
        'K': [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
    }
    outp.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--no_full_frame', dest='full_frame', action='store_false', help='Disable full-frame overlay rendering')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--save_structured', dest='save_structured', action='store_true', default=True,
                        help='Save structured predictions (MANO, keypoints, camera) to JSON (default: enabled)')
    parser.add_argument('--no_save_structured', dest='save_structured', action='store_false',
                        help='Disable structured JSON export')
    parser.add_argument('--structured_file', type=str, default='hamer_structured_results.json',
                        help='Structured output JSON filename under out_folder')
    parser.add_argument('--structured_debug', dest='structured_debug', action='store_true', default=False,
                        help='If set, save additional debug fields for validation')
    parser.add_argument('--camera_fx', type=float, default=None, help='Override camera fx in pixels for structured projection')
    parser.add_argument('--camera_fy', type=float, default=None, help='Override camera fy in pixels for structured projection')
    parser.add_argument('--camera_cx', type=float, default=None, help='Override camera cx in pixels for structured projection')
    parser.add_argument('--camera_cy', type=float, default=None, help='Override camera cy in pixels for structured projection')
    parser.add_argument('--intrinsics_file', type=str, default=None,
                        help='Optional path to a text file containing fx/fy/cx/cy (e.g. "Camera1.fx: 511.88"). '
                             'If set, overrides --camera_fx/fy/cx/cy unless those are explicitly provided.')
    parser.add_argument('--camera_json_out', type=str, default=None,
                        help='If set, write the effective intrinsics (fx/fy/cx/cy) to this JSON path. '
                             'Default: write to hamer/camera.json when --intrinsics_file is provided.')
    parser.add_argument('--assume_fps', type=float, default=None,
                        help='If set, add timestamp_sec = frame_idx / assume_fps for image sequences')
    parser.add_argument('--cam2base_json', type=str, default=None,
                        help='Optional path to JSON with 4x4 "T_cam2base" (camera->robot base) to export p_wrist_base / R_wrist_base')
    parser.add_argument('--temporal_smoothing', dest='temporal_smoothing', action='store_true', default=True,
                        help='Enable temporal smoothing for video sequences (default: enabled)')
    parser.add_argument('--no_temporal_smoothing', dest='temporal_smoothing', action='store_false',
                        help='Disable temporal smoothing')
    parser.add_argument('--smoothing_alpha', type=float, default=0.3,
                        help='Smoothing factor for exponential moving average (0-1, lower=smoother, default=0.3)')
    parser.add_argument('--min_det_score', type=float, default=0.5,
                        help='Minimum detection score to include in smoothing (default=0.5)')

    args = parser.parse_args()

    intr_from_file: Optional[Dict[str, float]] = None
    if args.intrinsics_file is not None:
        intr_from_file = _parse_intrinsics_file(args.intrinsics_file)
        # Only override CLI values if they were not explicitly provided.
        if args.camera_fx is None:
            args.camera_fx = intr_from_file['fx']
        if args.camera_fy is None:
            args.camera_fy = intr_from_file['fy']
        if args.camera_cx is None:
            args.camera_cx = intr_from_file['cx']
        if args.camera_cy is None:
            args.camera_cy = intr_from_file['cy']

        camera_json_out = args.camera_json_out
        if camera_json_out is None:
            camera_json_out = str((Path(__file__).resolve().parent / 'camera.json'))
        _write_camera_json(camera_json_out, args.camera_fx, args.camera_fy, args.camera_cx, args.camera_cy)
        print(f'Wrote camera intrinsics to {camera_json_out}')

    T_cam2base = None
    if args.cam2base_json is not None:
        with open(args.cam2base_json, 'r', encoding='utf-8') as f:
            _extr = json.load(f)
        T_cam2base = np.asarray(_extr['T_cam2base'], dtype=np.float64)
        if T_cam2base.shape != (4, 4):
            raise ValueError('cam2base JSON must contain T_cam2base as a 4x4 matrix')
    
    # Initialize temporal smoother
    smoother = None
    if args.temporal_smoothing:
        try:
            from scipy.spatial.transform import Rotation as R_scipy
            smoother = TemporalSmoother(alpha=args.smoothing_alpha, min_score=args.min_det_score)
            print(f'Temporal smoothing enabled (alpha={args.smoothing_alpha}, min_score={args.min_det_score})')
        except ImportError:
            print('Warning: scipy not available, disabling temporal smoothing')
            args.temporal_smoothing = False

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths = sorted(img_paths)
    structured_results = []

    # Iterate over all images in folder
    for frame_idx, img_path in enumerate(img_paths):
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        hand_scores = []
        hand_keypoints_2d = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                hand_scores.append(float(keyp[valid, 2].mean()))
                hand_keypoints_2d.append(left_hand_keyp.copy())
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                hand_scores.append(float(keyp[valid, 2].mean()))
                hand_keypoints_2d.append(right_hand_keyp.copy())

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            img_w = float(img_size[0, 0].item())
            img_h = float(img_size[0, 1].item())
            default_focal = float(model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max().item())
            fx = float(args.camera_fx) if args.camera_fx is not None else default_focal
            fy = float(args.camera_fy) if args.camera_fy is not None else default_focal
            cx = float(args.camera_cx) if args.camera_cx is not None else img_w / 2.0
            cy = float(args.camera_cy) if args.camera_cy is not None else img_h / 2.0
            focal_for_cam_t = float((fx + fy) * 0.5)
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_for_cam_t).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                keypoints_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                keypoints_3d[:,0] = (2*is_right-1)*keypoints_3d[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                if args.save_structured:
                    verts_cam = verts + cam_t[None, :]
                    keypoints_3d_cam = keypoints_3d + cam_t[None, :]
                    pred_mano = out['pred_mano_params']
                    global_orient = pred_mano['global_orient'][n].detach().cpu().numpy()
                    hand_pose = pred_mano['hand_pose'][n].detach().cpu().numpy()
                    betas = pred_mano['betas'][n].detach().cpu().numpy()
                    hand_det_id = int(person_id)
                    person_score = None
                    if hand_det_id < len(hand_scores):
                        person_score = float(hand_scores[hand_det_id])
                    
                    # Wrist pose in camera frame (meters / HaMeR weak-perspective scale); R from MANO global_orient.
                    p_wrist_cam = np.asarray(keypoints_3d_cam[0], dtype=np.float64)
                    R_wrist_cam = np.asarray(global_orient[0], dtype=np.float64)
                    hand_side = 'right' if int(is_right) == 1 else 'left'
                    
                    # Apply temporal smoothing
                    if smoother is not None:
                        p_wrist_cam, R_wrist_cam, cam_t_smooth, global_orient_smooth, hand_pose_smooth, betas_smooth = \
                            smoother.update(
                                p_wrist_cam, R_wrist_cam, cam_t,
                                global_orient, hand_pose, betas,
                                score=person_score
                            )
                        # Use smoothed values
                        cam_t = cam_t_smooth
                        global_orient = global_orient_smooth
                        hand_pose = hand_pose_smooth
                        betas = betas_smooth
                        # Update keypoints based on smoothed wrist position
                        wrist_delta = p_wrist_cam - keypoints_3d_cam[0]
                        keypoints_3d_cam = keypoints_3d_cam + wrist_delta[None, :]
                        keypoints_3d = keypoints_3d_cam - cam_t[None, :]
                    
                    # Convert to IsaacLab-compatible format with wrist in device/base frame
                    p_wrist_base = p_wrist_cam.copy()
                    R_wrist_base = R_wrist_cam.copy()
                    if T_cam2base is not None:
                        p_wrist_base, R_wrist_base = _wrist_pose_cam_to_base(
                            p_wrist_cam, R_wrist_cam, T_cam2base
                        )
                    
                    # keypoints_3d_local: OpenPose 21 keypoints in wrist-local frame
                    keypoints_3d_local = keypoints_3d.tolist()
                    
                    # Build minimal IsaacLab-compatible record
                    record = {
                        'frame_idx': int(frame_idx),
                        'hand_side': hand_side,
                        'score': person_score if person_score is not None else 1.0,
                        'p_wrist_base': p_wrist_base.tolist(),
                        'R_wrist_base': R_wrist_base.tolist(),
                        'keypoints_3d_local': keypoints_3d_local,
                    }
                    structured_results.append(record)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=focal_for_cam_t,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

    if args.save_structured:
        structured_path = os.path.join(args.out_folder, args.structured_file)
        # Wrap in "frames" key for IsaacLab compatibility (matching HOT3D adapter format)
        output_data = {'frames': structured_results}
        with open(structured_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f'Saved structured predictions to {structured_path} ({len(structured_results)} frames)')

if __name__ == '__main__':
    main()
