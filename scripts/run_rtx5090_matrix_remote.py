#!/usr/bin/env python3
"""RTX 5090 Daytona allocation and matrix preflight entrypoint.

This script reuses the shared RTX 5090 helper so the project has a single
deterministic sandbox-creation path built on `GpuType.RTX_5090`.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

from daytona import Daytona

from daytona_rtx5090_probe import build_params, build_daytona_config, describe_params


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "daytona-gpu" / "step10-rtx5090-feature-matrix"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="Create the sandbox.")
    parser.add_argument("--target", default=None, help="Optional Daytona target/region placement string.")
    parser.add_argument("--api-url", default=None, help="Optional Daytona API URL override.")
    parser.add_argument("--api-key", default=None, help="Optional Daytona API key override.")
    args = parser.parse_args()

    params = build_params()
    payload = describe_params(params)

    print(f"python: {sys.executable}")
    print(f"daytona: {metadata.version('daytona')}")
    print(f"resources repr: {repr(params.resources)}")
    print(f"params repr: {repr(params)}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    resources_json = payload.get("resources", {})
    print(f"serialized gpu={resources_json.get('gpu')}")
    print(f"serialized gpuType={json.dumps(resources_json.get('gpuType'))}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sandbox_request_payload.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.create:
        return 0

    daytona = Daytona(build_daytona_config(api_url=args.api_url, api_key=args.api_key, target=args.target))
    sandbox = daytona.create(params, timeout=600)
    metadata_obj = {
        "sandbox_id": sandbox.id,
        "state": str(sandbox.state),
        "sandbox": sandbox.model_dump(mode="json", by_alias=True, exclude_none=True),
    }
    (RESULTS / "sandbox_create_result.json").write_text(json.dumps(metadata_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata_obj, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
