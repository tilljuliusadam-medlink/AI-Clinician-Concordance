"""
Central main execution script to run all analyses.
"""

import time

import expert_study
import forest_plot
import regression
import tables_descriptive

STEPS = [("Table 1 and Supplementary Table 23", tables_descriptive.main),
         ("Tables 2-3 and Supplementary Tables 1-18, 20-21", regression.main),
         ("Table 4: expert-rated appropriateness", expert_study.main),
         ("Figure 1: forest plot", forest_plot.main)]


def main():
    for name, step in STEPS:
        print(f"\n=== {name} ===")
        started = time.perf_counter()
        step()
        print(f"--- {name} finished in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
