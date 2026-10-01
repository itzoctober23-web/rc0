#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fresh TorchScript (+ ONNX) export of a checkpoint's weights (clean process, contiguous tensors):
   python export_best.py best.ckpt best.pt   -> loads in ~3 s in the C++ engine (in-trainer traces took 12 s)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ZeroNet, net_for, in_planes_of, export_torchscript, export_onnx
if sys.argv[1].endswith(".pt"):  # a TorchScript export (e.g. an in-trainer trace that loads slowly): re-trace its weights
    sd = torch.jit.load(sys.argv[1], map_location="cpu").state_dict()
    blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")}) or 10
    ck = {"net": sd, "blocks": blocks, "filters": sd["stem.0.weight"].shape[0], "step": "?"}
else:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
net = net_for(ck["net"], ck.get("blocks", 10), ck.get("filters", 128))
net.load_state_dict({k: v.contiguous() for k, v in ck["net"].items()})
tmp = sys.argv[2] + ".tmp"
export_torchscript(net.eval(), tmp)
os.replace(tmp, sys.argv[2])
if sys.argv[2].endswith(".pt"):   # the portable copy every client downloads (ONNX Runtime backend)
    onnx = sys.argv[2][:-3] + ".onnx"
    export_onnx(net, onnx + ".tmp")
    os.replace(onnx + ".tmp", onnx)
print(f"exported step {ck.get('step')} -> {sys.argv[2]}")
