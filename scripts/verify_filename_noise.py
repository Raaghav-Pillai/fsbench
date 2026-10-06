"""Verify actual files in a paired filename-only sweep before spending API credits."""

import argparse
import json
from pathlib import Path

from fsbench.integrity import verify


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
