"""`python -m heron` is the same as the `heron` command."""
import sys

from heron.cli import main

sys.exit(main() or 0)
