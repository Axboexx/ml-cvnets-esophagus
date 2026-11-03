# 食管＋肝脏（＋脾脏）Grad-CAM

支持 `esophagus,liver` 和 `esophagus,liver,spleen` 两种 residual-context 模型，器官顺序必须与训练一致；三器官时把下例 overrides 改为 `esophagus,liver,spleen` / `15,5,1` / `0,0,0`、`num_organs=3`。图中每个器官一行，NPZ 键名为 `<organ>_cam`、`<organ>_ct`。

入口：`scripts/visualize_esophagus_liver_gradcam.py`。默认处理
`08887988 08577671 13646226 02377541 12123738`，保留前导零。

## 服务器运行

使用已有训练环境（PyTorch、NumPy、Matplotlib 及 ml-cvnets 依赖），在
`ml-cvnets/` 下运行。将下面的路径替换为服务器实际路径。
若 SuPreM 不在相邻目录，先设置 `SUPREM_ROOT` 为 SuPreM 仓库路径。

```bash
PYTHONHASHSEED=42 python scripts/visualize_esophagus_liver_gradcam.py \
  --config /path/to/matching_training_config.yaml \
  --checkpoint /path/to/internal_validation_auc_selected_checkpoint.pt \
  --data-root /path/to/youyi \
  --output-dir /path/to/new_gradcam_output \
  --device cuda \
  --override \
    dataset.e3a_multi_organ_roi_3d.organ_names=esophagus,liver \
    dataset.e3a_multi_organ_roi_3d.label_ids=15,5 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_xy=0,0 \
    dataset.e3a_multi_organ_roi_3d.roi_margin_z=0,0 \
    dataset.e3a_multi_organ_roi_3d.required_organs=esophagus,liver \
    model.classification.suprem_unet_e3a_residual_context.num_organs=2
```

上例的 overrides 对应 F3 两器官、零 margin 设置。必须与实际训练运行一致，
训练时修改过的 crop 高度、输入尺寸、窗宽窗位、context_scale 等也要在此恢复。
脚本不自动猜测 checkpoint，也不按外部 AUC 选模型。
请先从该食管＋肝脏运行的原始 `validation_confusion_matrix.jsonl`
（`stage=validation`）确认最高内部验证 AUC 对应 epoch，指定其保存的权重；
不要直接用按 top1 保存的 `checkpoint_best.pt`，也不要用 AUC 权重平均文件
冒充最高 AUC 单 epoch。若该 epoch 权重不存在，应明确这一限制。
完整训练 checkpoint 的 `model_state_dict`、`state_dict` 包装和裸 state dict
均支持，加载采用 strict 模式，缺失权重会报错。无需额外的 SuPreM 预训练权重。

数据目录沿用外部评估：

```text
youyi/
  youyi-nii/{hr,nhr}/{病例编号}/CT.nii.gz
  youyi-mask/{hr,nhr}/{病例编号}/mask.nii.gz
```

病例编号必须与目录名精确对应。同一编号跨类别重复、病例缺失、CT/mask 不对齐、
食管高度不满足训练配置、任一器官缺失都会报错。只读取指定病例的影像像素。
输出目录必须为新目录，避免覆盖已有图像。

## 方法与输出

- 默认解释 **HR 类别 pre-softmax logit**，通过 `--target-class 0` 可解释 no-HR。
- 对同一次食管＋肝脏联合预测求梯度，目标层默认
  `backbone.down_tr512.ops`，分别保留两个器官的 feature/gradient。
  3D 空间平均梯度加权 feature，再 ReLU；不改训练权重或推理归一化行为。
- 默认每器官 3 张轴位切片，按 mask 面积从大到小、加切片间隔自动选取，
  与预测和 CAM 无关；可用 `--slices-per-organ` 调整。
- 左侧为完整 CT 轴位截面，经训练窗宽窗位显示；右侧在同一截面叠加 ROI 热力图。
  CT 不做模型输入尺寸的拉伸，CAM 上采样回原始 ROI 网格后按实际裁剪坐标回填。
  ROI 外不着色，不能将 ROI 模型的 CAM 解释为全截面注意力。
  保留原始像素纵横比；图上 z 是原始 CT 体积内的零起始层索引。
  使用 native voxel 轴位，显示时在原方向基础上上下翻转（CT 与热力图同步），
  未进行标准解剖重定向，不标注 L/R。
- `case_01.png/.pdf` 等：600 dpi PNG 和 PDF，编号按输入顺序。
  图中不显示医院病例编号。图像内部的 CT 和热力图均为栅格。
- `case_01.npz` 等：各器官的原始 ROI CT（HU）与归一化 3D CAM。
- `metadata.json`：病例映射、真实标签、当前前向 HR 概率、切片索引、
  spacing、ROI 裁剪坐标、ROI 内及完整 CT 切片索引、显示 HU 范围、原始 CAM
  最大值、空 CAM 标记、配置/权重 SHA256、解析后的参数。
  该文件和 NPZ 包含病例关联或影像信息，论文只取匿名图像面板。
- `environment_packages.txt`：已安装包的版本清单；Python、CUDA 和 GPU 信息见 metadata。

CAM 在每个器官的整个 3D 体积内按最大值归一化到 [0,1]，切片共用范围。
图中色条代表相对强度，不能比较器官或病例之间的贡献大小。
叠加使用 jet 蓝—青—黄—红色阶与固定透明度（`--alpha`，默认 0.55），
低值和零值区域显示蓝色，保留完整 ROI 内的非负 CAM 分布。
蓝色不表示高关注，也不表示负贡献；整个器官 CAM 全零时只显示原 CT。
零 CAM 保持全零并记录，不人为增强或制造热点。
CAM 是模型 logit 的粗粒度解释，不能视为病灶分割或因果证据。
具体实现依据 [Grad-CAM 原论文](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)。

**批量依赖**：SuPreM 的 `ContBatchNorm3d.forward()` 在 eval 下仍以
`training=True` 运行。脚本忠实保留这一行为，逐病例前向同时输入两个器官；
因此概率和 CAM 可能随评估批量组成变化，不能声称复现原批量评估概率。
论文应注明可视化 batch size=1；必要时与服务器原始预测核对差异。

当前尚无服务器 checkpoint 和 CT，不能确认权重确为最高内部 AUC，
也未完成真实病例的数值与图像验收。服务器运行后应检查 metadata 状态为
`complete`、5 组病例映射和切片、空间对齐、色条、空 CAM 与 PDF 排版。
