#!/usr/bin/env python3
"""Verify a staged reference package with the production Python interpreter."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    reference = args.release.resolve() / "reference"

    required = (
        "worker.py",
        "manifest.json",
        "requirements.txt",
        "requirements-lock.txt",
        "constraints-cpu.txt",
        "tsys/main.py",
        "tsys/config/config.yaml",
    )
    missing = [name for name in required if not (reference / name).is_file()]
    if missing:
        parser.exit(1, f"参考发布包缺少文件：{reference} 下的 {', '.join(missing)}\n")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference))
    try:
        from worker import engine_digest, resolve_runtime_profile

        digest = engine_digest()
        profile = resolve_runtime_profile("source_default")
    except Exception as error:
        parser.exit(1, f"参考发布包校验失败：{error}\n")
    print(f"参考发布包校验通过，源码摘要 {digest}，运行档位 {profile}")


if __name__ == "__main__":
    main()
