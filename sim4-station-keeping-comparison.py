"""Generate the Simulation 4 raw-feedback versus EKF station-keeping PDFs.

Run from the project directory:
    python sim4-station-keeping-comparison.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_case_part_2 as runner
from simulation.plotter_part_2 import save_station_keeping_comparison


def main():
    logs = []
    for use_observer, label in ((False, "without observer"), (True, "with EKF")):
        print(f"Running Simulation 4 station keeping {label} (600 s)...", flush=True)
        cfg = runner.Part2SimConfig(dt=runner.DT, T=600.0,
                                    use_observer=use_observer, observer_type="kalman",
                                    use_sensor_noise=False)
        logs.append(runner.run(cfg, np.zeros(6), runner.environment(seed=runner.SEED)))
    paths = save_station_keeping_comparison(
        logs[0], logs[1], Path(__file__).resolve().parent / "output" / "pdf" / "sim4"
    )
    for path in paths:
        print(f"Saved {path}", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
