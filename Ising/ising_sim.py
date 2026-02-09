# ising_sim.py
import numpy as np
import random
import os
import argparse
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, CancelledError
from ising_initialization import ensure_open_dir


_stop_requested_event = threading.Event()
_current_mp_stop_event = None


def request_stop():
    global _current_mp_stop_event
    _stop_requested_event.set()
    if _current_mp_stop_event is not None:
        _current_mp_stop_event.set()


def clear_stop_request():
    global _current_mp_stop_event
    _stop_requested_event.clear()
    _current_mp_stop_event = None


def is_stop_requested():
    return _stop_requested_event.is_set()




# ================================
# ISING MODEL
# ================================
class IsingModel3D:
    def __init__(self, H, W, L, start_T, target_T, dT, steps_dT, j_coupling=1.0, external_field=0.0):
        self.H = H
        self.W = W
        self.L = L
        self.N = H * W * L

        self.temperature = start_T
        self.target_T = target_T
        self.dT = dT
        self.steps_dT = steps_dT

        self.j_coupling = j_coupling
        self.external_field = external_field

        self.spins = np.random.choice([-1, 1], size=(H, W, L)).astype(np.int8)
        self.total_spin = np.sum(self.spins)

        self.energy = self.compute_total_energy()
        self.magnetization = self.total_spin / self.N

    def compute_total_energy(self):
        E = 0.0
        for i in range(self.H):
            for j in range(self.W):
                for k in range(self.L):
                    E += self.get_neighbors_energy(i, j, k)
        return E / 2.0  # divide by 2 because each pair counted twice

    def get_neighbors_energy(self, i, j, k):
        spin = self.spins[i, j, k]
        neighbors_sum = 0
        for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            ni = (i + di) % self.H
            nj = (j + dj) % self.W
            nk = (k + dk) % self.L
            neighbors_sum += self.spins[ni, nj, nk]
        return -self.j_coupling * spin * neighbors_sum - self.external_field * spin

    def metropolis_step(self):
        i = random.randint(0, self.H - 1)
        j = random.randint(0, self.W - 1)
        k = random.randint(0, self.L - 1)

        s = self.spins[i, j, k]
        dE = -2 * self.get_neighbors_energy(i, j, k)

        if dE <= 0 or random.random() < np.exp(-dE / self.temperature):
            self.spins[i, j, k] *= -1
            self.energy += dE
            self.total_spin += -2 * s
            self.magnetization = self.total_spin / self.N

    def run_steps(self, num_steps):
        for _ in range(num_steps):
            for _ in range(self.N):  # N = H*W*L
                self.metropolis_step()

# -------------------------------
# Function to run one temperature
# -------------------------------
def run_temp(T, args, stop_event=None):
    model = IsingModel3D(args.H, args.W, args.L, start_T=T, target_T=args.target_T,
                         dT=args.dT, steps_dT=args.steps_dT, j_coupling=args.J, external_field=args.B)
    total_steps = int(args.steps * (1 + args.relax_perc / 100.0))
    relax_steps = total_steps - args.steps
    snapshots = []

    for step in range(1, total_steps + 1):
        if stop_event is not None and stop_event.is_set():
            break

        model.run_steps(1)

        if stop_event is not None and stop_event.is_set():
            break

        if step > relax_steps:
            if args.dT != 0.0 and model.temperature != model.target_T and args.steps_dT > 0 and step % args.steps_dT == 0:
                model.temperature += args.dT
                if (args.dT > 0 and model.temperature > args.target_T) or (args.dT < 0 and model.temperature < args.target_T):
                    model.temperature = args.target_T

            if step % args.save == 0:
                snapshot = {
                    "step": step - relax_steps,
                    "spins": model.spins.copy(),
                    "energy": model.energy,
                    "magnetization": model.magnetization,
                    "temperature": model.temperature
                }
                snapshots.append(snapshot)

                if args.stop_if_no_change == "ON" and len(snapshots) >= args.nochange_X:
                    recent_mags = [s['magnetization'] for s in snapshots[-args.nochange_X:]]
                    avg_dM = np.mean(np.abs(np.diff(recent_mags)))
                    if avg_dM < args.dM_limit:
                        print(f"T={T:.4f} stopped early: reached steady state (avg_dM={avg_dM:.2e}).")
                        break

                if args.stop_fully_mag == "ON" and abs(model.magnetization) >= args.fully_mag_limit:
                    stop_sim = True
                    print(f"T={T:.4f} stopped early: fully magnetized (|M| >= {args.fully_mag_limit}).")
                    break

    # Save snapshots to temp file
    temp_save_path = os.path.join(args.output, f"snapshots_T={T:.2f}.npz")
    np.savez_compressed(temp_save_path, snapshots=snapshots, H=args.H, W=args.W, L=args.L)
    return T, snapshots

# -------------------------------
# MAIN SIMULATION
# -------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="3D Ising Model Simulation")
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--W", type=int, default=20)
    parser.add_argument("--L", type=int, default=20)

    parser.add_argument("--start_T", type=float, default=1.0)
    parser.add_argument("--target_T", type=float, default=1.0)
    parser.add_argument("--dT", type=float, default=0.0)
    parser.add_argument("--steps_dT", type=int, default=1)

    parser.add_argument("--multi_temp", type=str, choices=["ON","OFF"], default="OFF")
    parser.add_argument("--min_T", type=float, default=5.0)
    parser.add_argument("--max_T", type=float, default=2.0)
    parser.add_argument("--num_T", type=int, default=4)

    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--relax_perc", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)

    parser.add_argument("--stop_if_no_change", type=str, choices=["ON","OFF"], default="OFF")
    parser.add_argument("--dM_limit", type=float, default=1e-5)
    parser.add_argument("--nochange_X", type=int, default=5)

    parser.add_argument("--stop_fully_mag", type=str, choices=["ON","OFF"], default="OFF")
    parser.add_argument("--fully_mag_limit", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="ising_sim")

    return parser


def run_simulation(args, progress_callback=None, stop_requested=None):
    args.output = ensure_open_dir(args.output)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.multi_temp == "ON" and args.num_T > 1:
        temperatures = np.linspace(args.min_T, args.max_T, args.num_T)
    else:
        temperatures = [args.start_T]

    n_workers = os.cpu_count() or 1
    print(f"Using {n_workers} CPU threads")
    if progress_callback is not None:
        progress_callback("workers", n_workers)

    all_snapshots = {}
    stopped_early = False

    clear_stop_request()

    # Run temps in parallel
    with multiprocessing.Manager() as manager:
        mp_stop_event = manager.Event()

        global _current_mp_stop_event
        _current_mp_stop_event = mp_stop_event

        with ProcessPoolExecutor() as executor:
            pending = {executor.submit(run_temp, T, args, mp_stop_event): T for T in temperatures}

            while pending:
                if is_stop_requested() or (stop_requested is not None and stop_requested()):
                    request_stop()
                    stopped_early = True

                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)

                for future in done:
                    try:
                        T, snapshots = future.result()
                    except CancelledError:
                        continue

                    all_snapshots[str(T)] = snapshots
                    print(f"Finished T={T:.4f}")
                    if progress_callback is not None:
                        progress_callback("finished_temp", T)

                if is_stop_requested() and pending:
                    for future in list(pending):
                        future.cancel()

            executor.shutdown(wait=False, cancel_futures=True)

    _current_mp_stop_event = None

    # Combine all temps into final file
    save_path = os.path.join(args.output, "snapshots.npz")
    save_dict = {str(temp): snaps for temp, snaps in all_snapshots.items()}

    # Add metadata
    save_dict.update({
        'H': args.H, 'W': args.W, 'L': args.L,
        'J': args.J, 'B': args.B,
        'start_T': args.start_T, 'target_T': args.target_T, 'dT': args.dT,
        'steps_dT': args.steps_dT,
        'steps': args.steps, 'relax_perc': args.relax_perc,
        'save_every': args.save,
        'stop_if_no_change': args.stop_if_no_change,
        'dM_limit': args.dM_limit,
        'nochange_X': args.nochange_X,
        'stop_fully_mag': args.stop_fully_mag,
        'fully_mag_limit': args.fully_mag_limit,
        'seed': args.seed,
        'stopped_early': stopped_early
    })

    np.savez_compressed(save_path, **save_dict)
    if stopped_early:
        print(f"\nSimulation stopped early. Partial snapshots saved to '{save_path}'.")
    else:
        print(f"\nAll temperatures complete. Combined snapshots saved to '{save_path}'.")
    return save_path


def run_from_params(params, progress_callback=None, stop_requested=None):
    parser = build_parser()
    cli_args = [f"{key}={value}" for key, value in params.items()]
    args = parser.parse_args(cli_args)
    return run_simulation(args, progress_callback=progress_callback, stop_requested=stop_requested)


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_simulation(args)

if __name__ == "__main__":
    main()
