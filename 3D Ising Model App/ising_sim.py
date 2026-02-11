# ising_sim.py

import numpy as np
import random
import os
import argparse
import multiprocessing
import threading
import hashlib
import pickle
import json
import re
import uuid
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, CancelledError
from ising_initialization import ensure_open_dir

_stop_requested_event = threading.Event()
_current_mp_stop_event = None

_RUN_CACHE_PREFIX = "ss_"
_INDEX_FILENAME = "simulation_index.jsonl"

def _fmt_float(value):
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)

def _fmt_value(value):
    if isinstance(value, float):
        return _fmt_float(value)
    return str(value)

def _normalize_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

def _sig_value(value):
    value = _normalize_value(value)
    if isinstance(value, float):
        return repr(float(value))
    return repr(value)

def _build_run_items(args, override_start_T=None, force_single=False):
    start_T = args.start_T if override_start_T is None else override_start_T
    multi_flag = "OFF" if force_single else args.multi_temp

    items = [
        ("H", _normalize_value(args.H)),
        ("W", _normalize_value(args.W)),
        ("L", _normalize_value(args.L)),
        ("start_T", _normalize_value(start_T)),
        ("target_T", _normalize_value(args.target_T)),
        ("dT", _normalize_value(args.dT)),
        ("steps_dT", _normalize_value(args.steps_dT)),
        ("steps", _normalize_value(args.steps)),
        ("relax_perc", _normalize_value(args.relax_perc)),
        ("save", _normalize_value(args.save)),
        ("J", _normalize_value(args.J)),
        ("B", _normalize_value(args.B)),
        ("multi_temp", multi_flag),
    ]
    if multi_flag == "ON":
        items.extend([
            ("min_T", _normalize_value(args.min_T)),
            ("max_T", _normalize_value(args.max_T)),
            ("num_T", _normalize_value(args.num_T)),
        ])
    items.extend([
        ("stop_if_no_change", args.stop_if_no_change),
        ("dM_limit", _normalize_value(args.dM_limit)),
        ("nochange_X", _normalize_value(args.nochange_X)),
        ("stop_fully_mag", args.stop_fully_mag),
        ("fully_mag_limit", _normalize_value(args.fully_mag_limit)),
        ("seed", args.seed if args.seed is not None else "None"),
    ])
    return items

def build_run_key(args):
    items = _build_run_items(args)
    signature = "|".join(f"{k}={_sig_value(v)}" for k, v in items)
    run_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    description = ", ".join(f"{k}={_fmt_value(v)}" for k, v in items)
    return run_id, signature, description

def build_temp_key(args, temp):
    items = _build_run_items(args, override_start_T=_normalize_value(temp), force_single=True)
    signature = "|".join(f"{k}={_sig_value(v)}" for k, v in items)
    run_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return run_id, signature

def _unique_suffix():
    return uuid.uuid4().hex[:6]


def _cache_path(args, run_id):
    return os.path.join(args.output, f"{_RUN_CACHE_PREFIX}{run_id}.pkl")

def _index_path(args):
    return os.path.join(args.output, _INDEX_FILENAME)

def _ensure_index_entry(index_path, entry):
    existing_ids = set()
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_id = rec.get("id")
                if rec_id:
                    existing_ids.add(rec_id)
    if entry["id"] in existing_ids:
        return
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")

def _load_index_entries(index_path):
    entries = []
    if not os.path.exists(index_path):
        return entries
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(rec)
    return entries

def _normalize_signature(signature):
    parts = signature.split("|")
    normalized = []
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        value = value.strip()
        match = re.match(r"np\.(float\d+|int\d+)\((.+)\)", value)
        if match:
            value = match.group(2)
        normalized.append(f"{key}={value}")
    return "|".join(normalized)

def _save_cache(cache_path, snapshots_by_temp, metadata):
    payload = {
        "snapshots_by_temp": snapshots_by_temp,
        "metadata": metadata,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

def _load_cache(cache_path):
    with open(cache_path, "rb") as f:
        return pickle.load(f)

def _write_npz(save_path, snapshots_by_temp, metadata):
    save_dict = {str(temp): snaps for temp, snaps in snapshots_by_temp.items()}
    save_dict.update(metadata)
    np.savez_compressed(save_path, **save_dict)


def load_cached_run(cache_path):
    return _load_cache(cache_path)

def save_combined_npz(save_path, snapshots_by_temp, metadata):
    _write_npz(save_path, snapshots_by_temp, metadata)

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
    total_steps = args.steps
    relax_steps = int(total_steps * args.relax_perc / 100.0)
    snapshots = []

    for step in range(1, total_steps + 1):
        if stop_event is not None and stop_event.is_set():
            break

        model.run_steps(1)

        if stop_event is not None and stop_event.is_set():
            break

        if args.dT != 0.0 and model.temperature != model.target_T and args.steps_dT > 0 and step % args.steps_dT == 0:
                model.temperature += args.dT
                if (args.dT > 0 and model.temperature > args.target_T) or (args.dT < 0 and model.temperature < args.target_T):
                    model.temperature = args.target_T

        if step > relax_steps:

            if step % args.save == 0:
                snapshot = {
                    "step": step,
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
                    print(f"T={T:.4f} stopped early: fully magnetized (|M| >= {args.fully_mag_limit}).")
                    break

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

    index_path = _index_path(args)
    save_path = os.path.join(args.output, "snapshots.npz")

    metadata_base = {
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
        'seed': args.seed
    }
    if args.multi_temp == "ON" and args.num_T > 1:
        temperatures = [float(t) for t in np.linspace(args.min_T, args.max_T, args.num_T)]
    else:
        temperatures = [float(args.start_T)]

    n_workers = os.cpu_count() or 1
    print(f"Using {n_workers} CPU threads")
    if progress_callback is not None:
        progress_callback("workers", n_workers)

    all_snapshots = {}
    stopped_early = False

    def temp_key(temp):
        return str(float(temp))

    cache_enabled = args.seed is not None
    cached_keys = set()

    if cache_enabled:
        index_entries = _load_index_entries(index_path)
        legacy_sig_map = {}
        for rec in index_entries:
            if rec.get("type") != "temp":
                continue
            sig = rec.get("signature")
            cache_path = rec.get("cache_path")
            if sig and cache_path:
                legacy_sig_map[_normalize_signature(sig)] = cache_path

        for T in temperatures:
            temp_id, temp_sig = build_temp_key(args, T)
            temp_cache_path = _cache_path(args, temp_id)
            legacy_cache_path = legacy_sig_map.get(_normalize_signature(temp_sig))
            used_legacy_entry = False
            if not os.path.exists(temp_cache_path):
                if legacy_cache_path and os.path.exists(legacy_cache_path):
                    temp_cache_path = legacy_cache_path
                    used_legacy_entry = True
                else:
                    continue
            try:
                cached = _load_cache(temp_cache_path)
            except Exception:
                continue
            snapshots_by_temp = cached.get("snapshots_by_temp", {})
            key = temp_key(T)
            if key in snapshots_by_temp:
                all_snapshots[key] = snapshots_by_temp[key]
                cached_keys.add(key)

                cached_meta = cached.get("metadata", {}) or metadata_base
                if cached.get("metadata") in (None, {}, []):
                    _save_cache(temp_cache_path, {key: snapshots_by_temp[key]}, cached_meta)
                if not used_legacy_entry:
                    temp_desc = (
                        f"T={_fmt_value(T)}, H={_fmt_value(args.H)}, W={_fmt_value(args.W)}, L={_fmt_value(args.L)}, "
                        f"J={_fmt_value(args.J)}, B={_fmt_value(args.B)}, steps={_fmt_value(args.steps)}, "
                        f"seed={_fmt_value(args.seed if args.seed is not None else 'None')}"
                    )
                    _ensure_index_entry(index_path, {
                        "id": temp_id,
                        "type": "temp",
                        "temp": float(T),
                        "signature": temp_sig,
                        "description": temp_desc,
                        "cache_path": temp_cache_path,
                    })

    temps_to_run = [T for T in temperatures if temp_key(T) not in cached_keys]

    clear_stop_request()

    if temps_to_run:
        # Run temps in parallel
        with multiprocessing.Manager() as manager:
            mp_stop_event = manager.Event()

            global _current_mp_stop_event
            _current_mp_stop_event = mp_stop_event

            with ProcessPoolExecutor() as executor:
                pending = {executor.submit(run_temp, T, args, mp_stop_event): T for T in temps_to_run}

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

                        all_snapshots[temp_key(T)] = snapshots
                        print(f"Finished T={T:.4f}")
                        if progress_callback is not None:
                            progress_callback("finished_temp", T)

                        temp_id, temp_sig = build_temp_key(args, T)
                        temp_entry_id = temp_id
                        if cache_enabled:
                            temp_cache_path = _cache_path(args, temp_id)
                            if not os.path.exists(temp_cache_path):
                                _save_cache(temp_cache_path, {temp_key(T): snapshots}, metadata_base)
                        else:
                            temp_entry_id = f"{temp_id}_{_unique_suffix()}"
                            temp_cache_path = _cache_path(args, temp_entry_id)
                            _save_cache(temp_cache_path, {temp_key(T): snapshots}, metadata_base)
                        temp_desc = (
                            f"T={_fmt_value(T)}, "
                            f"dT={_fmt_value(args.dT)}, steps_dT={_fmt_value(args.steps_dT)}, "
                            f"H={_fmt_value(args.H)}, W={_fmt_value(args.W)}, L={_fmt_value(args.L)}, "
                            f"J={_fmt_value(args.J)}, B={_fmt_value(args.B)}, "
                            f"steps={_fmt_value(args.steps)}, "
                            f"seed={_fmt_value(args.seed if args.seed is not None else 'None')}"
                        )
                        _ensure_index_entry(index_path, {
                            "id": temp_entry_id,
                            "type": "temp",
                            "temp": float(T),
                            "signature": temp_sig,
                            "description": temp_desc,
                            "cache_path": temp_cache_path,
                        })

                    if is_stop_requested() and pending:
                        for future in list(pending):
                            future.cancel()

                executor.shutdown(wait=False, cancel_futures=True)

        _current_mp_stop_event = None
    else:
        print("All temperatures found in cache. Skipping simulation.")

    # Combine all temps into final file
    save_dict = {str(temp): snaps for temp, snaps in all_snapshots.items()}

    # Add metadata
    metadata = {
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
    }

    _write_npz(save_path, save_dict, metadata)
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
