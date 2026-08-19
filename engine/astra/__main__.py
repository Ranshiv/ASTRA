"""Engine entry point. Rust spawns this as a sidecar process.

`python -m astra`          serve JSON-lines RPC on stdin/stdout
`python -m astra --probe`  print a one-shot readiness report and exit
"""

from __future__ import annotations

import argparse
import json
import sys

from . import cache, config, hardware, logging_config, rpc


def probe() -> dict:
    """One-shot readiness report, used by the installer and by `npm run probe`."""
    paths = config.PATHS
    return {
        "version": rpc.PROTOCOL_VERSION,
        "root": str(paths.root),
        "device": hardware.select_device().to_dict(),
        "cache": cache.measure().to_dict(),
        "library_cache_binding": config.bind_library_caches(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="astra", description="ASTRA science engine")
    parser.add_argument("--probe", action="store_true",
                        help="print a readiness report as JSON and exit")
    args = parser.parse_args(argv)

    config.apply_cache_redirects()
    logging_config.configure()

    if args.probe:
        json.dump(probe(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    config.bind_library_caches()
    rpc.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
