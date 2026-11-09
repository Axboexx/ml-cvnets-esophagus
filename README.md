# 食管胃静脉曲张风险预测（基于 ml-cvnets）

本仓库是食管胃静脉曲张风险研究的训练代码，基于 Apple 开源的 [ml-cvnets](https://github.com/apple/ml-cvnets)（CVNets 0.4）修改而来。我们沿用了它的训练引擎、模型和数据集注册机制、配置系统以及 `cvnets-train` 入口，在此基础上加入了 3D CT 的器官 ROI 数据集、SuPreM 预训练骨干、多器官和临床融合模型，以及验证和统计分析脚本。

- **第一期（已完成）**：高危食管胃静脉曲张二分类（HR / no-HR）。同济医院数据用于训练和内部验证，友谊医院数据用于外部验证。
- **第二期（规划中）**：二级预防后的再出血风险预测（生存分析）。相关代码尚未加入本仓库。

> 本仓库不包含任何病例数据、影像、mask 或模型权重。

## 安装

建议使用 Python 3.10+ 和 PyTorch ≥ 1.12。

```bash
git clone https://github.com/Axboexx/ml-cvnets-esophagus.git
cd ml-cvnets-esophagus

conda create -n cvnets python=3.10.8
conda activate cvnets

pip install -r requirements.txt -c constraints.txt
pip install --editable .
```

SuPreM 系列模型还需要另外下载 SuPreM 的源码和预训练权重 `supervised_suprem_unet_2100.pth`，查找路径见 [项目文档](projects/esophagus_risk/README.md#suprem-u-net-roi-fine-tuning)。

## 训练示例

在仓库根目录运行：

```bash
cvnets-train \
  --common.config-file config/classification/esophagus_risk/suprem_unet_roi.yaml \
  --common.results-loc results/G1 \
  --common.override-kwargs \
    dataset.root_train=/path/to/dataset_root \
    model.classification.suprem_unet_roi.pretrained_path=/path/to/supervised_suprem_unet_2100.pth
```

数据目录结构、友谊外部验证的用法以及各系列的完整命令，见 [projects/esophagus_risk/README.md](projects/esophagus_risk/README.md)。

## 实验系列

当前使用的是 **G 系列**：每个器官只保留 mask 内的 CT，mask 外的体素在 CT 归一化之后填为 -1，每个器官只输入一个 CT 通道。F 系列及更早的实验保留了 mask 外的体素，其 checkpoint 和阈值不适用于 G 系列。

| 模型 | 内容 | 模型文件（`cvnets/models/classification/`） |
|---|---|---|
| G1 | 食管 ROI，SuPreM U-Net 微调 | `suprem_unet_roi.py` |
| G2a / G2b | 食管分 4 段的 MIL / 双分支 MIL | `suprem_unet_mil_roi.py` / `suprem_unet_dual_mil_roi.py` |
| G3 | 食管 + 肝 / 脾 / 门静脉的器官组合（residual-context） | `suprem_unet_e3a_residual_context.py` |
| G4 | 在 G3 基础上晚期融合 17 项临床指标 | `suprem_unet_g4_clinical_context.py` |

G 系列协议的完整说明（加窗、裁剪、checkpoint 选择规则、注意事项）见 [项目文档](projects/esophagus_risk/README.md#g-series-organ-masked-ct-protocol-2026-10-14)。

## 目录说明

相对原版 ml-cvnets，本项目新增或修改的主要部分：

| 路径 | 内容 |
|---|---|
| `projects/esophagus_risk/` | 项目文档、临床数据接口、ROI 工具、Grad-CAM |
| `data/datasets/classification/esophagus_risk_*.py` | 同济和友谊的 3D ROI 数据集（单器官、多器官、临床融合） |
| `cvnets/models/classification/suprem_*.py`、`esophagus_roi_3d.py` | SuPreM 骨干和各系列模型 |
| `config/classification/esophagus_risk/` | 各实验的 YAML 配置 |
| `engine/` | 增加了验证 AUC、混淆矩阵、逐例预测导出和按 AUC 保存 checkpoint |
| `scripts/` | 结果汇总、DeLong 检验、ROC / 校准曲线、Grad-CAM 可视化 |

每次训练输出的指标文件（`training_metrics.jsonl`、`validation_confusion_matrix.jsonl`、`training_run_summary.json` 等）见 [项目文档](projects/esophagus_risk/README.md#saved-training-metrics)。

## 许可与致谢

本仓库基于 Apple Inc. 的 [ml-cvnets](https://github.com/apple/ml-cvnets)，原始代码的版权和许可条款见 [LICENSE](LICENSE)，第三方组件见 [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS)。

如果使用了 CVNets，请引用：

```bibtex
@inproceedings{mehta2022cvnets,
  author    = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad},
  title     = {CVNets: High Performance Library for Computer Vision},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  series    = {MM '22},
  year      = {2022}
}
```

SuPreM 预训练模型来自 [MrGiovanni/SuPreM](https://github.com/MrGiovanni/SuPreM)，使用时请按其仓库要求引用。
