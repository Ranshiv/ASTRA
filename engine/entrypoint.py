"""PyInstaller entry point; keeps the astra package import semantics intact."""
from astra.__main__ import main

raise SystemExit(main())
