"""Convert the Jiang et al. camera sensitivity database for Harvard/ICVL.

Input curves cover 400-720 nm at 10 nm intervals. Harvard and ICVL use
31 bands from 420-720 nm, so the first two samples are removed. Each RGB
response is then normalized to sum to one, matching the project's RGB
synthesis convention.
"""

from pathlib import Path
import re

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SRF/test/source/camspec_database.txt"
OUTPUT = ROOT / "SRF/test"


def safe_name(camera_name):
    name = re.sub(r"[^a-zA-Z0-9]+", "_", camera_name.strip()).strip("_")
    return name.lower()


def load_database(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 4:
        raise ValueError(f"Expected groups of four lines, got {len(lines)} lines")

    cameras = []
    for offset in range(0, len(lines), 4):
        name = lines[offset]
        curves = np.asarray(
            [[float(value) for value in lines[offset + channel].split()]
             for channel in range(1, 4)],
            dtype=np.float64,
        )
        if curves.shape != (3, 33):
            raise ValueError(f"{name}: expected (3, 33), got {curves.shape}")
        cameras.append((name, curves))
    return cameras


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cameras = load_database(SOURCE)
    manifest = [
        "# Jiang et al. real camera SRFs converted for Harvard/ICVL",
        "# Source wavelengths: 400-720 nm, 10 nm interval",
        "# Output wavelengths: 420-720 nm, 10 nm interval",
        "# MAT variable: P_20N, shape [3,31], RGB rows sum to 1",
        "",
    ]

    for camera_name, curves in cameras:
        curves = curves[:, 2:]
        curves /= curves.sum(axis=1, keepdims=True)
        filename = f"{safe_name(camera_name)}_harvard.mat"
        savemat(OUTPUT / filename, {"P_20N": curves})
        manifest.append(f"{filename}\t{camera_name}")

    (OUTPUT / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"Converted {len(cameras)} camera SRFs into {OUTPUT}")


if __name__ == "__main__":
    main()
