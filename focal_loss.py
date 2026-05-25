from import_library import *

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=3.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        cw = torch.tensor([1.0, self.pos_weight], device=logits.device)
        ce = F.cross_entropy(logits, target, weight=cw, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = torch.where(
            target == 1,
            torch.full_like(ce, self.alpha),
            torch.full_like(ce, 1.0 - self.alpha)
        )
        loss = alpha_t * (1.0 - pt) ** self.gamma * ce
        return loss.mean()