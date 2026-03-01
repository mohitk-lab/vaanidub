#!/usr/bin/env python3
"""Download all required AI models for VaaniDub."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vaanidub.config import AppConfig
from vaanidub.models.model_manager import ModelManager, MODEL_REGISTRY


async def main():
    parser = argparse.ArgumentParser(description="Download VaaniDub AI models")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--model", type=str, help="Download specific model")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    config = AppConfig()
    manager = ModelManager()

    if args.list:
        print("\nAvailable models:")
        print("-" * 70)
        for name, info in MODEL_REGISTRY.items():
            print(f"  {name:<35} {info.download_size_gb:>5.1f} GB  {info.description}")
        print(f"\n  Total: {manager.get_total_download_size():.1f} GB")

        gpu = manager.check_gpu()
        if gpu.get("available"):
            print(f"\n  GPU: {gpu['device_name']} ({gpu['free_vram_mb']}MB free)")
        else:
            print(f"\n  GPU: Not available ({gpu.get('reason', 'unknown')})")

        reqs = manager.get_gpu_requirements()
        print(f"  {reqs['recommendation']}")
        return

    if args.all:
        print(f"Downloading all models ({manager.get_total_download_size():.1f} GB total)...")
        print(f"HuggingFace token: {'set' if config.hf_token else 'NOT SET (needed for gated models)'}")
        print()
        await manager.download_all(hf_token=config.hf_token)
        print("\nAll models downloaded successfully!")

    elif args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)
        print(f"Downloading {args.model}...")
        await manager.download_model(args.model, hf_token=config.hf_token)
        print(f"{args.model} downloaded successfully!")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
