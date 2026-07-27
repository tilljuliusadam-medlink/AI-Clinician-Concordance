"""Build every output table from the simulated data in data/.

    python main.py
"""

import time

import expert_study
import regression
import tables_descriptive


STEPS = [("Tables 1 and 2", tables_descriptive.main),
         ("Tables 3 and 4 plus Supplementary Tables 1 to 10", regression.main),
         ("expert-rating sub-study", expert_study.main)]


def main():
    for name, step in STEPS:
        print(f"\n=== {name} ===")
        started = time.perf_counter()
        step()
        print(f"--- {name} finished in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
