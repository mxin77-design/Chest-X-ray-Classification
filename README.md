# Chest X-ray Classification: Normal vs Abnormal

基于深度学习的胸部 X 光影像二分类课程项目。项目从零实现 CNN 基线，并将 ImageNet 预训练的 ResNet-18 迁移到 `abnormal` / `normal` 分类任务；最后使用验证集表现更好的模型为无标签测试图像生成预测。

> 这是用于课程研究与模型比较的实验，不是经过临床验证的医疗产品，也不能替代专业医师的判断。

## 项目背景与目标

医学影像分类可以帮助研究者探索深度学习从影像中提取特征的能力。本项目使用带有 `abnormal` 和 `normal` 标签的胸部 X 光影像，比较两种建模方案，并完成以下工作：

1. 从零训练一个卷积神经网络作为基线。
2. 微调 ImageNet 预训练的 ResNet-18，并与基线比较。
3. 使用验证集表现较好的模型对 100 张无标签测试图像进行预测。

## 数据集

| 数据划分 | Abnormal | Normal | 总数 |
|---|---:|---:|---:|
| 训练集 | 3,165 | 1,092 | 4,257 |
| 验证集 | 734 | 257 | 991 |
| 测试集 | 无标签 | 无标签 | 100 |

训练集存在类别不平衡：`abnormal` 类约为 `normal` 类的 2.9 倍。项目使用加权交叉熵提高少数类在训练损失中的权重。原始影像数据没有包含在此仓库中；运行 notebook 前，请自行准备有权使用的数据，并按下方目录结构放置。

## 技术与方法

- **语言与环境：** Python、Jupyter Notebook / Google Colab
- **深度学习：** PyTorch、torchvision
- **数据处理与评估：** NumPy、pandas、Pillow、scikit-learn
- **可视化：** Matplotlib、Seaborn
- **图像预处理：** 转为 RGB，缩放至 `224 × 224`，使用 ImageNet 均值和标准差归一化
- **训练增强：** 随机水平翻转、最多 10° 的随机旋转
- **优化：** 加权交叉熵、Adam（初始学习率 `1e-4`、weight decay `1e-4`）、基于验证准确率的 `ReduceLROnPlateau` 调度器
- **实验设置：** 随机种子 `42`，每个模型训练 10 个 epoch

### 模型

**自建 CNN** 包含四个卷积块，每块由卷积、Batch Normalization、ReLU 和最大池化组成；通道数从 32 增至 256，分类头使用全连接层和 Dropout。模型约有 2,608 万个可训练参数。

**ResNet-18** 使用 torchvision 提供的 ImageNet 预训练权重。训练时冻结前面的层，只微调 `layer4` 和替换后的二分类全连接层。模型共约 1,118 万个参数，其中约 839 万个可训练。

## 实验结果

下表结果来自 notebook 中的 **991 张验证图像**。AP 为 `abnormal` 类的 Average Precision；Precision / Recall / F1 按类别列出。

| 模型 | 验证准确率 | Abnormal P / R / F1 | Normal P / R / F1 | Abnormal AP | 验证集误分类数 |
|---|---:|---|---|---:|---:|
| 自建 CNN | 97.78% | 0.98 / 0.99 / 0.99 | 0.96 / 0.95 / 0.96 | 0.9992 | 22 |
| ResNet-18 微调 | **98.79%** | 0.99 / 0.99 / 0.99 | 0.98 / 0.97 / 0.98 | **0.9997** | **12** |

根据这组验证结果，ResNet-18 被选为最终模型。它的验证准确率更高，验证集误分类数量更少，同时可训练参数也少于自建 CNN。

在 100 张无标签测试图像上，ResNet-18 预测了 98 张 `abnormal`、2 张 `normal`。这是模型预测分布，不是有真实标签验证过的测试集准确率；测试标签不可用，因此无法由此计算测试性能。

## 推理代码示例

下面的示例加载 notebook 训练出的 ResNet-18 权重，对单张影像输出预测类别和模型置信度。请先将 `best_q2_resnet18.pth` 放在当前目录，或修改 `CHECKPOINT` 路径。

```python
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = Path("best_q2_resnet18.pth")
CLASS_NAMES = ["abnormal", "normal"]  # 与 ImageFolder 的类别索引顺序一致

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

image = Image.open("example.jpeg").convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

with torch.inference_mode():
    probabilities = model(input_tensor).softmax(dim=1)[0]

predicted_index = int(probabilities.argmax())
print("Prediction:", CLASS_NAMES[predicted_index])
print("Confidence:", float(probabilities[predicted_index]))
```

## 运行项目

主要实验代码位于 [`final.ipynb`](final.ipynb)，可在 Google Colab 中打开并运行。Notebook 默认从 Google Drive 读取数据，路径为 `/content/gdrive/MyDrive/mth416/data`；如使用其他位置，请修改 notebook 中的 `DATA_DIR`。

数据目录需符合 `torchvision.datasets.ImageFolder` 的格式：

```text
data/
├── train/
│   ├── abnormal/
│   └── normal/
├── val/
│   ├── abnormal/
│   └── normal/
└── test/
    └── test samples/
```

依赖可通过以下命令安装（Colab 通常已预装 PyTorch）：

```bash
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn seaborn
```

运行 notebook 后会训练两个模型、生成验证集评估结果，并将测试集预测保存为 CSV。默认保存位置在 Google Drive 的 `mth416` 目录，包括：

- `data/best_q1.pth`：自建 CNN 的最佳验证集权重
- `data/best_q2_resnet18.pth`：ResNet-18 的最佳验证集权重
- `q3_test_predictions_resnet18.csv`：包含预测类别和置信度
- `q3_test_predictions_for_report.csv`：仅包含图像 ID 和预测类别

## 仓库内容

- `final.ipynb`：数据加载、训练、评估和测试集预测的完整 notebook
- `sample_images.png`：训练集影像示例图
- `Final_Report.pdf`、`Final_PreSlides.pdf`、`Pre.pptx`：课程报告与展示材料

## 注意事项

- 仓库不包含原始影像数据或训练权重；请在公开上传前确认所有数据和展示材料都可以公开分享，并遵守数据集本身的许可与隐私要求。
- 测试集没有标签，测试集预测数量不能视为测试准确率。
- 结果来自单次课程实验设置；模型可能受到数据来源、类别不平衡和训练样本范围影响，尚未经过外部验证。
- 当前仓库未声明软件许可证；如计划允许他人复用代码，请补充合适的开源许可证。
