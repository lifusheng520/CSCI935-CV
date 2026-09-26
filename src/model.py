"""MALG-MobileNetV2: equations (2)--(5) in CSCI935_A2.pdf."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MALG(nn.Module):
    def __init__(self, classes=5, pretrained=True, variant="malg", temperature=0.5,
                 quality_threshold=0.45, mask_dropout=0.2):
        super().__init__()
        self.variant = variant
        self.temperature = temperature
        self.quality_threshold = quality_threshold
        self.mask_dropout = mask_dropout
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v2(weights=weights).features
        # features[6]: 32 x 28 x 28; features[18]: 1280 x 7 x 7 at 224px.
        self.se = nn.Sequential(nn.Linear(1280, 80), nn.ReLU(), nn.Linear(80, 1280), nn.Sigmoid())
        self.scorer = nn.Conv2d(32, 1, 1, bias=False)
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.heads = nn.ModuleList([nn.Linear(1280, classes), nn.Linear(32, classes), nn.Linear(32, classes)])
        self.gate = nn.Sequential(nn.Linear(1280 + 32 + 32 + 2, 64), nn.ReLU(), nn.Linear(64, 3))
        self.finetune_from = len(self.backbone)

    def set_stage(self, finetune=False):
        self.finetune_from = 14 if finetune else len(self.backbone)
        for i, block in enumerate(self.backbone):
            block.requires_grad_(i >= self.finetune_from)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BN running statistics must not drift during head training.
        if mode:
            for block in self.backbone[:self.finetune_from]:
                block.eval()
        return self

    def forward(self, image, mask, quality):
        x = image
        for i, block in enumerate(self.backbone):
            x = block(x)
            if i == 6:
                shallow = x
        pooled = x.mean((2, 3))
        zg = pooled if self.variant == "baseline" else pooled * self.se(pooled)
        selection = torch.sigmoid((self.scorer(shallow) - self.threshold) / self.temperature)
        zl = (shallow * selection).sum((2, 3)) / selection.sum((2, 3)).clamp_min(1e-6)
        coverage = F.interpolate(mask, size=shallow.shape[-2:], mode="area")
        zm = (shallow * coverage).sum((2, 3)) / coverage.sum((2, 3)).clamp_min(1e-6)
        valid = (quality >= self.quality_threshold) & (mask.sum((1, 2, 3)) > 0)
        if self.training and self.mask_dropout:
            valid = valid & (torch.rand_like(quality) >= self.mask_dropout)
        if self.variant in ("baseline", "global_local"):
            valid = torch.zeros_like(valid)
        # Rejected masks must have no influence even through the gate inputs.
        zm = zm * valid[:, None]
        q = quality * valid
        available = torch.stack((torch.ones_like(valid), torch.ones_like(valid), valid), 1)
        if self.variant == "baseline":
            available[:, 1] = False
        logits = torch.stack([head(z) for head, z in zip(self.heads, (zg, zl, zm))], 1)
        gate_logits = self.gate(torch.cat((zg, zl, zm, q[:, None], valid[:, None].float()), 1))
        weights = gate_logits.masked_fill(~available, -torch.inf).softmax(1)
        # Log-space probability mixture, NOT cross entropy applied to probabilities.
        log_prob = torch.logsumexp(logits.log_softmax(-1) + weights.clamp_min(1e-30).log()[:, :, None]
                                   + (~available)[:, :, None] * -1e9, dim=1)
        return dict(log_prob=log_prob, logits=logits, weights=weights,
                    available=available, selection=selection, valid=valid)


def objective(out, labels, auxiliary=0.2, selection_penalty=0.01, density=0.25):
    main = F.nll_loss(out["log_prob"], labels)
    losses = torch.stack([F.cross_entropy(out["logits"][:, b], labels, reduction="none")
                          for b in range(3)], dim=1)
    aux = ((losses * out["available"]).sum(1) / out["available"].sum(1)).mean()
    penalty = ((out["selection"].mean((1, 2, 3)) - density) ** 2).mean()
    return main + auxiliary * aux + selection_penalty * penalty
