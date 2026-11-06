"""Run FPE through the shared NDRR command interface."""
import sys
from ..cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fpe", *sys.argv[1:]]))
