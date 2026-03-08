#!/usr/bin/env python3
"""Convert vae_bn_stats.pt (PyTorch) to vae_bn_stats.json for C++ loading.

Usage:
    python convert_bn_stats.py ./exported_flux2_klein/vae_bn_stats.pt

Outputs vae_bn_stats.json in the same directory.
"""
import json
import sys
from pathlib import Path

import torch


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <vae_bn_stats.pt>")
        sys.exit(1)

    pt_path = Path(sys.argv[1])
    stats = torch.load(pt_path, map_location="cpu", weights_only=True)

    out = {
        "running_mean": stats["running_mean"].tolist(),
        "running_var": stats["running_var"].tolist(),
    }

    json_path = pt_path.with_name("vae_bn_stats.json")
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {json_path}  ({len(out['running_mean'])} channels)")


if __name__ == "__main__":
    main()
