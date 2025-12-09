import torch

def classification_metrics(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.view(-1)
    targets = targets.view(-1).long()
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    correct = (preds == targets).sum().item()
    acc = correct / (targets.numel() + 1e-6)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1 = 0.0 if (targets == 1).sum().item() == 0 else 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def landmark_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    diff = (pred - gt)**2
    dist = torch.sqrt(diff.sum(-1) + 1e-8)
    masked_dist = dist * mask
    mean_dist = masked_dist.sum() / (mask.sum() + 1e-6)
    return {"nme": mean_dist.item()}

def pck_metric(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, threshold: float = 0.1):
    dist = torch.sqrt(((pred - gt)**2).sum(-1))
    hits = ((dist <= threshold) * mask).sum().item()
    total = mask.sum().item() + 1e-6
    return {"pck": hits / total}