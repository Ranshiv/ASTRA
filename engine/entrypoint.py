"""PyInstaller entry point; keeps the astra package import semantics intact."""
import multiprocessing

if __name__ == "__main__":
    # Frozen executables must call freeze_support() before anything else:
    # multiprocessing bootstraps each worker by re-invoking this same .exe
    # with a sentinel argument, and freeze_support() is what intercepts that
    # sentinel and runs the worker instead of the app's real main(). Without
    # this guard a worker "spawn" just re-runs the whole engine (blocking on
    # its own stdin), and the parent's pool.map(...) waits forever.
    multiprocessing.freeze_support()
    from astra.__main__ import main

    raise SystemExit(main())
