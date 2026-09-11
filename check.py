"""Run the project checks and print a PASS/FAIL report.

Usage:
    python check.py                  # Part 1 subsystem checks and Part 1, Simulations 1-4
    python check.py --fast           # Part 1 subsystem checks only (seconds)
    python check.py --part 2         # Part 2 subsystem and observer requirement checks

Part 1 full checks include closed-loop simulation checks.  Part 2 public
checks verify interfaces and explicit subsystem requirements; closed-loop
performance is evaluated through the submitted results and staff assessment.

The same checks run via ``pytest`` and in the demo notebooks
(``notebooks/part_1_demo.ipynb``, ``notebooks/part_2_demo.ipynb``, where every
simulation is also plotted); see ``simulation/checks.py`` and
``simulation/checks_part_2.py`` for what each check verifies.  The exit code
is non-zero unless everything passes.  The CI workflow
(``.github/workflows/check.yml``) runs ``pytest``, which reports unimplemented
subsystems as skipped instead of failed, so a fresh template starts green on
GitHub.
"""
import argparse
import sys

from simulation.checks import print_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="run only the fast subsystem checks")
    parser.add_argument("--part", type=int, choices=(1, 2), default=1,
                        help="which project part to check (default: 1)")
    args = parser.parse_args()
    if args.part == 2:
        from simulation.checks_part_2 import run_all
    else:
        from simulation.checks import run_all
    return 0 if print_report(run_all(fast_only=args.fast)) else 1


if __name__ == "__main__":
    sys.exit(main())
