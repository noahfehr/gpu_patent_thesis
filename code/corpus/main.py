#!/usr/bin/env python3
"""Corpus preparation pipeline

This script ties together the various preprocessing stages for a given
version of the corpus.  It is intentionally simple: the caller provides a
version prefix (e.g. ``v1`` or ``v2``) and the pipeline will

1. locate the corresponding CSV export under ``data/raw``
2. filter the file by the CPC codes of interest (delegates to
   :mod:`preprocessing.filter_cpc`) and
3. hand the resulting CSV off to the claims extraction pipeline which
   appends full patent claims and writes a ``*_processed.csv`` file in
   ``data/claims_added`` (see :mod:`lens_api.claims_pipeline`).

The prefix is configurable via a variable at the top of the file or via
command line arguments when executed as a script.
"""

import argparse
import os
import sys

from preprocessing.filter_cpc import filter_cpc_csv
from lens_api.claims_pipeline import create_claims_pipeline



version_prefix = None


def run_pipeline(prefix: str):
    """Run the full preprocessing + claims pipeline for a given prefix.

    Parameters
    ----------
    prefix : str
        Version prefix such as ``v1`` or ``v2``.  Used to build file paths.
    """

    data_dir = "data"
    raw_dir = os.path.join(data_dir, "raw")
    csv_path = os.path.join(raw_dir, f"{prefix}_lens_export.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"raw CSV not found: {csv_path}")

    print("=" * 60)
    print(f"CORPUS PREPROCESSING PIPELINE - {prefix}")
    print("=" * 60)

    # filter to the CPC codes we care about
    print("\nSTEP 1: filtering CPC codes")
    filter_cpc_csv(csv_path)

    # after filtering, hand off to the claims pipeline
    print("\nSTEP 2: running claims extraction pipeline")
    final_output = create_claims_pipeline(prefix)

    print("\nPipeline complete. Final file:\n", final_output)
    return final_output

def main():
    global version_prefix

    parser = argparse.ArgumentParser(
        description="Corpus preprocessing + claims pipeline."
    )
    parser.add_argument(
        "--prefix",
        help="version prefix (e.g. v1, v2) to operate on."
    )
    args = parser.parse_args()

    prefix = args.prefix or version_prefix
    if prefix is None:
        parser.error(
            "must specify a version prefix via --prefix or by "
            "setting the `version_prefix` variable in the file"
        )

    try:
        run_pipeline(prefix)
    except Exception as exc:
        print(f"pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
