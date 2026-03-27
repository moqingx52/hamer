# HaMeR 环境部署
## 一、项目简介
HaMeR 是基于单张图像/视频的**3D 手部网格重建模型**，输出 MANO 手部参数 + 3D 顶点，可直接用于手部动作数字化，是 Real2Sim（真实世界→仿真）的核心上游工具。

---

## 二、环境部署（完整可复制命令）
### 1. 克隆项目（含子模块）
```bash
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
```

### 2. 创建并激活 Conda 环境
```bash
conda create --name hamer python=3.10 -y
conda activate hamer
```

### 3. 安装 PyTorch（官方指定 CUDA 11.7）
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

### 4. 安装项目依赖 + ViTPose
```bash
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

### 5. 下载官方 Demo 数据
```bash
bash fetch_demo_data.sh
```

### 6. 关键依赖修复（必执行，解决版本冲突）
```bash
# 卸载冲突包 + 清空缓存
pip uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless xtcocotools
pip cache purge

# 安装指定版本依赖
pip install "numpy==1.23.5" "cython<3" "wheel" "setuptools<81"
pip install "opencv-python==4.8.1.78"
pip install --no-build-isolation --no-binary xtcocotools xtcocotools
```

---

## 三、运行命令（官方 Demo 全场景）
### 1. 基础示例运行
```bash
python demo.py \
  --img_folder example_data \
  --out_folder demo_out \
  --batch_size=48 \
  --side_view \
  --save_mesh \
  --full_frame
```

### 2. 更换人体检测器（RegNetY）
```bash
python demo.py \
  --img_folder example_data \
  --out_folder demo_out \
  --side_view \
  --save_mesh \
  --full_frame \
  --body_detector regnety
```

### 3. 视频转帧 + 批量处理（视频手部重建）
```bash
# 新建文件夹 + 视频抽帧
mkdir -p frames
ffmpeg -i input.mp4 -q:v 2 frames/%06d.jpg

# 运行重建
python demo.py \
  --img_folder frames \
  --out_folder demo_out \
  --file_type "*.jpg" \
  --side_view \
  --save_mesh \
  --full_frame \
  --body_detector regnety
```

---

## 四、核心功能：HaMeR 能输出什么？
### ✅ 官方 Demo 直接提供的能力
1. **人体检测 → 手部关键点检测（ViTPose）→ 手部框定位**
2. **3D 手部网格回归**（MANO 模型）
3. 输出文件：
   - 渲染图（原图叠加 3D 手部）
   - 3D 网格文件（`.obj`）
   - 核心模型输出：`pred_vertices`（3D顶点）、`pred_cam`（相机参数）、`pred_cam_t_full`（相机平移）
4. 支持**单张图/图片序列/视频**输入

---

## 五、Real2Sim 能力评估：还差什么？
### 1. HaMeR 已完成：**第一步（真实手部 → 3D 手部参数/网格）**
✅ 直接满足：**视频手部动作 → 仿真器 MANO 网格** 基础需求

### 2. 完整 Real2Sim 必须额外补充（官方 Demo 不提供）
1. **时序平滑/跟踪**（消除单帧抖动）
2. **相机内参标定 + 世界坐标系对齐**
3. **人体/物体位姿关联**
4. **导出仿真器可用的骨架控制参数**
5. 时序数据结构化存储

---

## 六、研究/二次开发必看：MANO 参数保存（核心！）
### 重要说明
- 仅保存 `.obj` 网格**无法用于后续仿真/重定向/SMPL-X**
- 必须保存 **MANO 原生参数**（官方明确支持）

### 最小修改代码（在 demo.py 中添加）
找到代码中 `out = model(batch)` 位置，添加以下代码，保存 `.npz` 格式参数：
```python
# 逐只手保存 MANO 参数 + 3D 信息
params = {
    "is_right": int(batch["right"][n].cpu().numpy()),  # 左右手标记
    "global_orient": out["pred_mano_params"]["global_orient"][n].detach().cpu().numpy(),  # 全局朝向
    "hand_pose": out["pred_mano_params"]["hand_pose"][n].detach().cpu().numpy(),  # 手指姿态
    "betas": out["pred_mano_params"]["betas"][n].detach().cpu().numpy(),  # 手部形状
    "pred_cam": out["pred_cam"][n].detach().cpu().numpy(),  # 相机参数
    "pred_cam_t_crop": out["pred_cam_t"][n].detach().cpu().numpy(),  # 裁剪图相机平移
    "pred_cam_t_full": pred_cam_t_full[n],  # 全图相机平移
    "pred_vertices": verts,  # 3D 顶点
}

# 保存文件
np.save(f"output_{n}.npz", params)
```

---

## 七、左手特殊注意事项
1. 官方 Demo 对**左手做镜像处理**（X 轴翻转）
2. 模型内部会将左手转换为右手格式处理
3. 接入仿真系统时：**左手必须手动还原坐标/姿态约定**，否则姿态完全错误

---

## 八、常用命令速查
```bash
# 激活环境
conda activate hamer
# 进入项目目录
cd /depot/gsy/external/hamer
# 视频抽帧
ffmpeg -i input.mp4 -q:v 2 frames/%06d.jpg
```

---

### 总结
1. **部署**：按文档命令一步执行，依赖修复是关键，可直接跑通官方 Demo
2. **功能**：HaMeR 是 3D 手部重建工具，能输出 MANO 参数 + 3D 网格
3. **Real2Sim**：已完成手部数字化核心步骤，缺时序、相机、仿真适配层
4. **开发**：务必保存 MANO 参数，左手注意镜像规则
