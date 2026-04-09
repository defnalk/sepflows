"""
Benchmark for the rigorous distillation MESH solver hot path.

RigorousColumn.solve runs an outer iteration where each pass (1) builds a
K-value matrix via n_stages separate k_values_raoult() calls, and (2) runs
a per-stage temperature update that calls k_values_raoult() again. Both of
those inner calls do per-component Python dict lookups into the ANTOINE
database and a list-comprehension over components — N * nc dict hits per
pass. For a 20-stage, 4-component column that's 160 Antoine lookups per
iteration, and convergence needs ~25-100 iterations.

# BASELINE (pre-optimization):
#   mean wall time over 5 runs (20 column solves, ~26 iter each): 0.1329 s
#   cProfile (cumulative):
#     solve                    0.198 s / 20 calls
#     k_values_raoult          0.053 s / 20,800 calls
#     antoine_pressure         0.011 s / 41,600 calls  (dict lookup per call)
#     _thomas_solve            0.038 s / 1,040 calls
#     logging.debug            0.019 s / 20,820 calls  (in k_values_raoult)
#
# AFTER optimization (vectorized _k_matrix with precomputed Antoine vectors):
#   mean wall time over 5 runs: 0.0597 s
#   speedup: 2.23x
#   k_values_raoult and antoine_pressure no longer on the hot path;
#   K-values now come from a single broadcast expression per iteration
#   instead of N*nc Python-level dict lookups.
#
# Verified: distillate composition identical to baseline (0.6000018, 0.3999982)
# for the methanol/water test case, 26 iterations, converged=True.
# All 126 tests pass unchanged.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from sepflows.distillation.rigorous import RigorousColumn

N_RUNS = 5


def workload() -> None:
    # Realistic column: methanol / water separation at atmospheric pressure.
    col = RigorousColumn(
        components=["methanol", "water"],
        n_stages=20,
        feed_stage=10,
        reflux_ratio=2.72,
        distillate_to_feed=0.55,
        pressure_pa=101_325.0,
        feed_temperature_k=337.0,
    )
    z = np.array([0.60, 0.40])
    # Run the solver 20 times to dominate over fixed init overhead.
    for _ in range(20):
        col.solve(feed_flow=65_000.0, z=z)


def main() -> None:
    workload()  # warm up

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        workload()
        times.append(time.perf_counter() - t0)

    mean = sum(times) / len(times)
    print(f"mean wall time over {N_RUNS} runs: {mean:.4f} s")
    print(f"individual runs: {['%.4f' % t for t in times]}")

    prof = cProfile.Profile()
    prof.enable()
    workload()
    prof.disable()
    s = StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(15)
    print("\ncProfile (top 15 by cumulative time):")
    print(s.getvalue())


if __name__ == "__main__":
    main()
