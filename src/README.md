# MALG-MobileNetV2：可运行 pipeline

代码对应 `ideas/CSCI935_A2.pdf` 第 4 节。所有命令在项目根目录执行。
数据默认读取 `Dhan-Shomadhan/`，不用移动或重命名原图。
依赖、预训练权重、训练输出都保留在本地；`tmp/refered code/` 不再是运行时依赖。

## 1. 安装与运行

作业要求 Python 3.12；建议新环境使用它。当前机器只有系统 Python 3.9，已用它建立 `.venv`
并做离线运行验证，尚未在 Python 3.12 上执行。torchvision 是 PyTorch 的视觉模型包；
gdown 仅用于下载 U²-Net 的官方权重。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements.txt
```

当前机器已安装好依赖，可直接 `source .venv/bin/activate`，无需重建环境。

先下载权重（需要网络；约 190 MB），再跑一轮完整训练：

```bash
python -m src.pipeline download --u2net
python -m src.pipeline run --field-masks u2net
```

默认 224×224、batch 16，先冻结 backbone 训练 5 epochs，再部分解冻训练最多 15 epochs。
自动选择 CUDA、Apple MPS 或 CPU。`--device cpu` 可强制 CPU；内存不足时减小 `--batch-size`。
第一轮会生成并缓存 Mask，后续直接复用。每个 epoch 输出 train loss、validation NLL 和 accuracy。

**先快速看是否能跑通**（仍使用预训练模型和 U²-Net，但只取 80 张原图，训练 2 epochs）：

```bash
python -m src.pipeline run --quick --field-masks u2net --head-epochs 1 --finetune-epochs 1
```

**完全离线验证**（无需权重，不代表真实分类效果）：

```bash
python -m src.pipeline run --quick --no-pretrained --field-masks none --head-epochs 1 --finetune-epochs 1 --device cpu
```

`--quick` 固定每个类别×背景取 train=4、val=2、test=2，总共 40/20/20 张。
`--no-pretrained` 使用随机 backbone，仅供调试；冻结随机特征的结果不用于判断方法表现。
普通运行默认 `--field-masks auto`：找到权重就使用，否则会明确禁用田间 Mask。
**正式实验用 `--field-masks u2net`**，权重缺失会报错，避免不知情地运行删减版。

## 2. 输出在哪里

默认保存到 `artifacts/runs/<时间>-malg/`。可以用 `--output 新目录` 指定位置；
已有目录不会覆盖。各 seed 子目录包含：

| 文件 | 内容 |
|---|---|
| `best.pt` | 验证 NLL 最佳模型、类别顺序、推理配置 |
| `config.json` / `split.json` | 超参数、环境版本、实际划分、图像 SHA256 |
| `history.json` / `training.png` | 训练曲线与阶段记录 |
| `metrics.json` | White、Field、Mixed 的 Accuracy、Macro-F1、每类指标、Mask 接受率 |
| `predictions.csv` | 每张测试图的标签、预测、五类概率、Mask 质量和三分支权重 |
| `confusion_*.png` | 三种测试背景的混淆矩阵 |
| `attention_masks.png` | 各背景×类别示例的原图、Mask、软选择热图、融合权重 |

run 根目录还有 `summary.json` 和 `mask_summary.json`。
注意力图表示模型空间选择，不能视为病斑定位真值；Mask 的 q 也不是分割准确率。

## 3. 对自己的图片预测

把下面的 `RUN` 和图片路径换成实际路径（预测输出目录须不存在）：

```bash
python -m src.pipeline predict \
  --checkpoint artifacts/runs/RUN/seed_42/best.pt \
  --image 'Dhan-Shomadhan/White Background/Brown Spot/bs_wb_0.jpg' \
  --background White \
  --output artifacts/prediction_example
```

输出 `prediction.json` 和 `prediction.png`。`--background` 支持 `White`、`Field`、`auto`；
auto 用图像边缘白色比例做简单路由，已知背景时建议显式指定。
模型只区分给定五种疾病，不包含健康或未知疾病类别。
移动项目后如 U²-Net 权重路径变化，可用 `--u2-weights 新路径` 覆盖。

## 4. 五次随机划分与对照实验

```bash
python -m src.pipeline run --field-masks u2net --seeds 42 43 44 45 46
python -m src.pipeline run --variant baseline --field-masks none --seeds 42 43 44 45 46
python -m src.pipeline run --variant global_local --field-masks none --seeds 42 43 44 45 46
```

- `malg`：PDF 中完整三分支方法。
- `baseline`：MobileNetV2 + GAP + linear，不使用 SE 或局部/Mask 融合。
- `global_local`：SE global + soft local，不使用 Mask 分支。

相同 seed 的原始划分完全一致；70%/15%/15% 按类别×背景联合分层，均混合背景训练，
在同一个 held-out test 的 White/Field 子集和整体分别评估。1106 张实际分成 774/166/166。
随机划分间测试样本可能重叠，这不是五折交叉验证。
标准差用样本标准差 `ddof=1`；单次运行记为 null。
测试集只在训练结束、恢复最佳验证 checkpoint 后评估。
解冻阶段从 Stage A 最佳权重开始；两阶段均用验证 NLL 做 early stopping，最终取全程最佳模型。
辅助 loss 和密度正则包含在训练目标中，validation NLL 只计算最终融合分类项，因此两条 loss 曲线量级不同。
baseline 为便于比较仍实例化公共模块；参数总数包含未使用分支，不用于声称 baseline 的部署效率。

## 5. 实现细节与设计对应

- `data.py`：修正目录别名 `Browon Spot → Brown Spot`、`Leaf Scaled → Leaf Scald`、
  `Rice Turgro → Rice Tungro`、`Shath Blight → Sheath Blight`，映射到固定五类。
  图像字节级重复会报错，避免跨集合重复；近似重复、同一植株相关图没有元数据可分组，仍需人工审查。
- `masks.py`：White 使用 HSV 近白背景取反，保留黄褐色组织，再闭运算、去小连通域、填孔。
  因实拍白纸偏灰，亮度/饱和度阈值根据每张图低饱和边缘像素的中位数自适应，避免整张图被当作前景。
  Field 使用冻结 U²-Net 的 fused saliency map，在 320×320 上推理，min-max 归一化后阈值 0.5。
  `q = area_score × (0.5 + 0.5 × largest_component_fraction) × mean(2|S−0.5|)`；
  `area_score = clip(min(area/0.05, (1−area)/0.10), 0, 1)`。
  默认 q≥0.45 且非空时接受。属于未校准启发式，需只用训练/验证样本人工检查和调参。
- `model.py`：MobileNetV2 `features[6]` 为 32×28×28 浅层，最终层为 1280×7×7。
  Global 为 SE+GAP；Local 按 PDF 式 (2) 使用可学习阈值的 sigmoid 权重池化，默认 τ=0.5。
  Mask 使用 area resize 计算覆盖率后池化。门控输入三组特征、q、v，并在 softmax 前屏蔽无效分支。
  无效 Mask 特征也清零，不会通过 gate 间接影响预测。最终混合概率以 log-space 计算 NLL。
- `pipeline.py`：头部 AdamW lr=1e-3；Stage B 解冻 features[14:]，backbone lr=1e-5，新增层 lr=1e-4；
  weight decay=1e-4；Mask dropout=0.2；λaux=0.2、λsel=0.01、ρ=0.25；patience=5。
  冻结 block 的 BatchNorm 统计保持冻结。图像和 Mask 同步水平翻转及 ±15° 旋转，
  图像单独轻微亮度/对比度变化，再做 ImageNet 标准化。验证/测试没有随机增强。
- `reporting.py`：逐图预测、混淆矩阵、注意力/Mask 可视化、五次运行汇总。

没有添加旧 `design.md` 中的 Top-k 或背景替换；最终 PDF 使用 soft selection，也未要求背景替换。
来源和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 6. 权重与缓存

MobileNetV2 自动下载到项目根目录的 `models/torch/checkpoints/`。
U²-Net 放在 `models/u2net.pth`，官方仓库及手动下载链接：

- [U²-Net 官方说明](https://github.com/xuebinqin/U-2-Net)
- [官方 u2net.pth](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view)

Google Drive 限流时，可浏览器下载后放到上述位置。
也支持轻量版 `download --u2net --u2-arch u2netp` 和 `run --u2-arch u2netp --field-masks u2net`，
这是不同的分割器配置，做实验时需单独注明。
Mask 缓存键包含分割器权重 SHA256、图像 SHA256、图像尺寸和预处理版本；改变这些输入会重新生成。
缓存与训练结果保存在项目根目录的 `artifacts/` 下；预训练权重在 `models/` 下。
`Dhan-Shomadhan/` 只保留原始数据集，不写入生成文件。不要将缓存打包进作业提交。

## 7. 验证

```bash
python -m unittest src.test_pipeline -v
```

检查软选择/门控反向传播、冻结 BN、无效 Mask 不影响输出、概率归一化、Mask dropout、
白色背景下棕色组织保留、分层划分互斥及可重复性。
目前实际执行的是 CPU、随机初始化、80 张子集、1+1 epochs 的离线验证。
预训练权重下载未获授权，因此完整预训练训练和真实 U²-Net 权重推理尚未验证。
离线 smoke 的可视化已用于开发检查，不应把这次 smoke 的测试结果当作独立正式实验结果。

本机已生成的验证文件：

- `artifacts/runs/smoke_verified/seed_42/attention_masks.png`
- `artifacts/runs/smoke_verified/seed_42/best.pt`
- `artifacts/prediction_verified/prediction.png`

该离线运行 test Accuracy=20%，Macro-F1≈0.0667，只说明随机初始化的调试模型没有学到有效分类。
U²-Net/U²-NetP 的 320×320 随机权重前向形状检查也已通过，但不能替代预训练分割质量验证。
