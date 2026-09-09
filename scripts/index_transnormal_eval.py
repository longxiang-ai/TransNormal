"""Index the public TransNormal tar shards using only the fixed 395 test views."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transnormal.training.data.index import create_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing uncompressed public .tar shards")
    parser.add_argument("--output", required=True, help="New JSONL index filename")
    args = parser.parse_args()
    print(create_index("transnormal", args.root, args.output, ROOT / "configs/splits/transnormal_test.txt", shards=True))
