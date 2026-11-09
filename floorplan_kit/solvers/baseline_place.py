"""Official baseline (score = 1.00 by definition). Deliberately NOT tuned.

Spring model trained with PyTorch:
  loss = wirelength (smooth |dx|+|dy|)  +  LAM * pairwise overlap area  +  MU * out-of-outline
  random init, Adam with a fixed learning rate, constant LAM/MU, fixed 300 steps.
Output goes to the official legalizer afterwards (the runner does that automatically).
Ignores all variant rules (forbidden / edge / fixed / rotate / distance).
"""
import torch

STEPS, LR, LAM, MU = 300, 0.01, 1.0, 10.0


def init(inst, W, H, w, h):
    return torch.rand(len(w)) * (W - w) + w / 2, torch.rand(len(h)) * (H - h) + h / 2


def loss_fn(x, y, w, h, a, b, wt, W, H, eps):
    wl = (wt * (torch.sqrt((x[a] - x[b]) ** 2 + eps ** 2) + torch.sqrt((y[a] - y[b]) ** 2 + eps ** 2))).sum()
    ox = torch.relu((w[:, None] + w[None]) / 2 - (x[:, None] - x[None]).abs())
    oy = torch.relu((h[:, None] + h[None]) / 2 - (y[:, None] - y[None]).abs())
    overlap = torch.triu(ox * oy, diagonal=1).sum()
    out = (torch.relu(w / 2 - x) + torch.relu(x + w / 2 - W) + torch.relu(h / 2 - y) + torch.relu(y + h / 2 - H)) ** 2
    return wl, overlap, out.sum()


def place(inst, time_limit):
    W0, H0 = inst["outline"]
    S = (W0 * H0) ** 0.5                                    # normalise so the outline is ~1 x 1
    W, H = W0 / S, H0 / S
    w = torch.tensor([b["w"] for b in inst["blocks"]], dtype=torch.float32) / S
    h = torch.tensor([b["h"] for b in inst["blocks"]], dtype=torch.float32) / S
    a = torch.tensor([e["a"] for e in inst["nets"]]); b = torch.tensor([e["b"] for e in inst["nets"]])
    wt = torch.tensor([float(e["w"]) for e in inst["nets"]])
    wt = wt / wt.sum()
    x, y = init(inst, W, H, w, h)
    x.requires_grad_(True); y.requires_grad_(True)
    opt = torch.optim.Adam([x, y], lr=LR)
    eps = 0.01 * float(w.mean())
    for _ in range(STEPS):
        opt.zero_grad()
        wl, ov, out = loss_fn(x, y, w, h, a, b, wt, W, H, eps)
        (wl + LAM * ov + MU * out).backward()
        opt.step()
    x, y = x.detach(), y.detach()
    return [[float((x[i] - w[i] / 2) * S), float((y[i] - h[i] / 2) * S)] for i in range(len(w))]
