#!/usr/bin/env python3
"""Compatibility entry point for the unified saved-prediction evaluator.

This wrapper replaces an archived script whose private helper import was not
part of the release. The unified evaluator reports CSA Pearson/Spearman
together with the other paper metrics from the same generated-cell file.
"""

from evaluate_saved_predictions import main


if __name__ == "__main__":
    main()
