#!/usr/bin/env python3
"""Compatibility entry point for the unified saved-prediction evaluator.

ERC is computed as overlap divided by the observed interval width. This
wrapper replaces an archived script whose model imports were not included in
the release. See ``evaluate_saved_predictions.py --help`` for the CLI.
"""

from evaluate_saved_predictions import main


if __name__ == "__main__":
    main()
