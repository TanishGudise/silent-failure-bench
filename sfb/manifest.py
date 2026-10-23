"""Hashes of the pre-registration and the code that defines the measurement.

`python -m sfb.manifest` writes manifest.json; `--check` fails if anything
changed since. Paid runs record the manifest hash they ran under.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ["preregistration.md", "pyproject.toml", "scripts/gateway.sh", "scripts/harness.sh",
          "tools/netguard/sitecustomize.py", "tools/netguard/loopback.sb"] + sorted(
    f"sfb/{p.name}" for p in (ROOT / "sfb").glob("*.py"))


def compute() -> dict:
    files = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"files": files, "manifest_hash": combined}


def main() -> None:
    current = compute()
    path = ROOT / "manifest.json"
    if "--check" in sys.argv:
        saved = json.loads(path.read_text())
        changed = [n for n in FROZEN if saved["files"].get(n) != current["files"][n]]
        print("manifest OK" if not changed else f"changed since freeze: {changed}")
        sys.exit(1 if changed else 0)
    path.write_text(json.dumps(current, indent=1) + "\n")
    print(current["manifest_hash"])


if __name__ == "__main__":
    main()
