# 实现来源

- `vendor/u2net.py` 来自用户提供的 `tmp/refered code/U-2-Net-master/model/u2net.py`。
  上游：[xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)。
  Apache-2.0 原始许可保留于 `vendor/U2NET_LICENSE`。仅添加出处说明，并将废弃的
  `F.upsample` 改成等价的 `F.interpolate(..., align_corners=False)`。
- MobileNetV2 直接调用 torchvision 实现及 `IMAGENET1K_V2` 权重，
  [官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)。
- 参考 `project-tomato-inference` 的 transfer learning、数据增强、最佳验证模型保存和结果导出流程。
  没有照搬其灰度 CLAHE：本方法需要保留黄色、棕色及绿色组织的颜色特征。
- 参考 `Rice-Leaf-Disease-Detection-main` 的 PyTorch 五分类组织方式。
  其自定义 CNN 和已有 `model.pth` 不用于这里的 MobileNetV2。
- MALG 三分支、soft selection、可靠性门控及损失按照 `ideas/CSCI935_A2.pdf` 第 4 节重新实现。
  这是项目设计的实现，并非 DBLA 或 A2QTrans 的完整复现。
