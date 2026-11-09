"""Sanity solver (no PyTorch needed): random positions, then the runner's legalizer fixes them.
Use it to check that your installation works before running the PyTorch baseline."""
import random


def place(inst, time_limit):
    W, H = inst["outline"]
    return [[random.uniform(0, W - b["w"]), random.uniform(0, H - b["h"])] for b in inst["blocks"]]
