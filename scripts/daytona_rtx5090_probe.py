#!/usr/bin/env python3
"""Daytona sandbox creation helper for deterministic RTX 5090 selection.

This module does not create a sandbox when imported. It exists so the project
can use the newer Daytona SDK surface (`GpuType` and `Resources.gpu_type`) for
RTX 5090 placement when the operator explicitly opts in.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, GpuType, Image, Resources


DEFAULT_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"


@dataclass(frozen=True)
class SandboxProbeSpec:
    name: str = "cuda-checkpoint-rtx5090-matrix"
    image: str = DEFAULT_IMAGE
    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None
    gpu_count: int = 1
    gpu_types: tuple[GpuType, ...] = (GpuType.RTX_5090,)
    env_vars: dict[str, str] | None = None
    labels: dict[str, str] | None = None
    public: bool | None = None
    ephemeral: bool = True
    auto_pause_interval: int | None = None
    auto_stop_interval: int = 0
    auto_archive_interval: int | None = None
    auto_delete_interval: int = 0
    secrets: dict[str, str] | None = None
    network_block_all: bool | None = None
    network_allow_list: str | None = None
    domain_allow_list: str | None = None
    linked_sandbox: str | None = None


def build_params(spec: SandboxProbeSpec = SandboxProbeSpec()) -> CreateSandboxFromImageParams:
    """Build the Daytona create params for an RTX 5090-backed sandbox."""

    return CreateSandboxFromImageParams(
        name=spec.name,
        image=Image.base(spec.image),
        cpu=spec.cpu,
        memory=spec.memory,
        disk=spec.disk,
        resources=Resources(gpu=spec.gpu_count, gpu_type=list(spec.gpu_types)),
        env_vars=spec.env_vars,
        labels=spec.labels,
        public=spec.public,
        ephemeral=spec.ephemeral,
        auto_pause_interval=spec.auto_pause_interval,
        auto_stop_interval=spec.auto_stop_interval,
        auto_archive_interval=spec.auto_archive_interval,
        auto_delete_interval=spec.auto_delete_interval,
        secrets=spec.secrets,
        network_block_all=spec.network_block_all,
        network_allow_list=spec.network_allow_list,
        domain_allow_list=spec.domain_allow_list,
        linked_sandbox=spec.linked_sandbox,
    )


def build_daytona_config(*, api_url: str | None = None, api_key: str | None = None, target: str | None = None) -> DaytonaConfig:
    """Build a Daytona client config without changing any sandbox state."""

    return DaytonaConfig(api_url=api_url, api_key=api_key, target=target)


def describe_params(params: CreateSandboxFromImageParams) -> dict[str, Any]:
    """Return a JSON-serializable payload preview."""

    resources = params.resources
    return {
        "name": params.name,
        "image": DEFAULT_IMAGE,
        "resources": {
            "gpu": resources.gpu if resources else None,
            "gpu_type": [gpu_type.value for gpu_type in resources.gpu_type] if resources and resources.gpu_type else None,
            "gpuType": [gpu_type.value for gpu_type in resources.gpu_type] if resources and resources.gpu_type else None,
        },
        "ephemeral": params.ephemeral,
        "auto_stop_interval": params.auto_stop_interval,
        "auto_delete_interval": params.auto_delete_interval,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="Create the sandbox instead of only printing the payload.")
    parser.add_argument("--target", default=None, help="Optional Daytona target/region placement string.")
    parser.add_argument("--api-url", default=None, help="Optional Daytona API URL override.")
    parser.add_argument("--api-key", default=None, help="Optional Daytona API key override.")
    args = parser.parse_args()

    spec = SandboxProbeSpec()
    params = build_params(spec)
    print(json.dumps(describe_params(params), indent=2, sort_keys=True))

    if not args.create:
        return 0

    config = build_daytona_config(api_url=args.api_url, api_key=args.api_key, target=args.target)
    daytona = Daytona(config)
    sandbox = daytona.create(params, timeout=600)
    print(json.dumps({"sandbox_id": sandbox.id, "state": str(sandbox.state)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
