# ising_plots.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import re
from ising_initialization import resolve_snapshot_file

sns.set_style("whitegrid")

# ------------------------
# Load snapshots
# ------------------------
def load_snapshots(file_path):
    """
    Load snapshots from a compressed .npz file.
    
    Returns:
        all_snapshots: dict mapping temperature -> list of snapshots
        params: dict of additional parameters (for single-temp runs)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    data = np.load(file_path, allow_pickle=True)

    def extract_temp(key):
        try:
            return float(key)
        except ValueError:
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(key))
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return None
        return None

    # Case 1: single-temperature run saved as 'snapshots'
    if 'snapshots' in data.files:
        snapshots = data['snapshots'].tolist()
        params = {}
        for k in data.files:
            if k == 'snapshots':
                continue
            arr = data[k]
            params[k] = arr.item() if arr.shape == () else arr
        temp_key = float(params['target_T']) if 'target_T' in params else 1.0
        return {temp_key: snapshots}, params

    # Case 2: multi-temperature run saved as temperature keys
    all_snapshots = {}
    temp_keys = set()
    for key in data.files:
        temp = extract_temp(key)
        if temp is None:
            continue
        temp_keys.add(key)
        if temp in all_snapshots:
            all_snapshots[temp].extend(data[key].tolist())
        else:
            all_snapshots[temp] = data[key].tolist()

    metadata = {}
    for key in data.files:
        if key in temp_keys:
            continue
        arr = data[key]
        metadata[key] = arr.item() if arr.shape == () else arr

    return all_snapshots, metadata

def compute_min_energy(metadata):
    try:
        H = int(metadata.get("H"))
        W = int(metadata.get("W"))
        L = int(metadata.get("L"))
        J = float(metadata.get("J", 1.0))
        B = float(metadata.get("B", 0.0))
    except (TypeError, ValueError):
        return None

    N = H * W * L
    e_up = -3.0 * J * N - B * N
    e_down = -3.0 * J * N + B * N
    return min(e_up, e_down)

def split_by_step_resets(snapshots):
    """
    Split snapshots into segments when step counter resets or decreases.
    This avoids connecting separate runs in line plots.
    """
    segments = []
    current = []
    last_step = None
    for snap in snapshots:
        step = snap.get("step")
        if last_step is not None and step is not None and step <= last_step:
            if current:
                segments.append(current)
            current = []
        current.append(snap)
        last_step = step
    if current:
        segments.append(current)
    return segments

def temperature_varies(snapshots, tol=1e-9):
    temps = [s.get("temperature") for s in snapshots if "temperature" in s]
    if len(temps) < 2:
        return False
    return (max(temps) - min(temps)) > tol


# ------------------------
# Time evolution plots
# ------------------------

def plot_energy_time(file_path):
    all_snapshots, metadata = load_snapshots(file_path)
    if not all_snapshots:
        print("No snapshots found.")
        return

    temps_sorted = sorted(all_snapshots.keys())
    cmap = cm.get_cmap("viridis", len(temps_sorted))  # gradient
    show_temp = any(temperature_varies(snaps) for snaps in all_snapshots.values() if snaps)
    e_min = compute_min_energy(metadata or {})
    norm_label = "Energy"
    if e_min is not None and e_min != 0:
        norm_label = "Energy / |E_min|"
    if show_temp:
        fig, (ax_energy, ax_temp) = plt.subplots(2, 1, figsize=(8,7), sharex=True)
    else:
        fig, ax_energy = plt.subplots(1, 1, figsize=(8,5))

    for i, temp in enumerate(temps_sorted):
        snaps = all_snapshots[temp]
        if not snaps:
            continue
        coupling = snaps[0].get("J", 1.0)  # default to 1.0 if not present
        external_field = snaps[0].get("B", 0.0)  # default to 0.0 if not present

        segments = split_by_step_resets(snaps)
        for j, segment in enumerate(segments):
            steps = [s["step"] for s in segment]
            energy = [s["energy"] for s in segment]
            if e_min is not None and e_min != 0:
                energy = [e / abs(e_min) for e in energy]
            label = f"T={temp:.3g}" if j == 0 else None
            ax_energy.plot(steps, energy, color=cmap(i), label=label)

        if show_temp:
            for j, segment in enumerate(segments):
                steps = [s["step"] for s in segment]
                temps = [s.get("temperature") for s in segment]
                ax_temp.plot(steps, temps, color=cmap(i))
    ax_energy.set_ylim(-1.05, 0.05)
    ax_energy.set_ylabel(norm_label)
    ax_energy.set_title("Energy vs Step")
    ax_energy.legend()
    ax_energy.grid(True)

    if show_temp:
        ax_temp.set_xlabel(f"MC Steps for J={coupling}, B={external_field}")
        ax_temp.set_ylabel("Temperature")
        ax_temp.set_title("Temperature vs Step")
        ax_temp.grid(True)
    else:
        ax_energy.set_xlabel(f"MC Steps for J={coupling}, B={external_field}")

    fig.tight_layout()
    plt.show()

def plot_mag_time(file_path):
    all_snapshots, _ = load_snapshots(file_path)
    if not all_snapshots:
        print("No snapshots found.")
        return

    temps_sorted = sorted(all_snapshots.keys())
    cmap = cm.get_cmap("viridis", len(temps_sorted))
    show_temp = any(temperature_varies(snaps) for snaps in all_snapshots.values() if snaps)
    if show_temp:
        fig, (ax_mag, ax_temp) = plt.subplots(2, 1, figsize=(8,7), sharex=True)
    else:
        fig, ax_mag = plt.subplots(1, 1, figsize=(8,5))

    for i, temp in enumerate(temps_sorted):
        snaps = all_snapshots[temp]
        if not snaps:
            continue
        coupling = snaps[0].get("J", 1.0)  # default to 1.0 if not present
        external_field = snaps[0].get("B", 0.0)  #

        segments = split_by_step_resets(snaps)
        for j, segment in enumerate(segments):
            steps = [s["step"] for s in segment]
            mag = [s["magnetization"] for s in segment]
            label = f"T={temp:.3g}" if j == 0 else None
            ax_mag.plot(steps, mag, color=cmap(i), label=label)

        if show_temp:
            for j, segment in enumerate(segments):
                steps = [s["step"] for s in segment]
                temps = [s.get("temperature") for s in segment]
                ax_temp.plot(steps, temps, color=cmap(i))

    ax_mag.set_ylim(-1.05, 1.05)
    ax_mag.set_ylabel("Magnetization")
    ax_mag.set_title("Magnetization vs Step")
    ax_mag.legend()
    ax_mag.grid(True)

    if show_temp:
        ax_temp.set_xlabel(f"MC Steps for J={coupling}, B={external_field}")
        ax_temp.set_ylabel("Temperature")
        ax_temp.set_title("Temperature vs Step")
        ax_temp.grid(True)
    else:
        ax_mag.set_xlabel(f"MC Steps for J={coupling}, B={external_field}")

    fig.tight_layout()
    plt.show()

# ------------------------
# Thermodynamic plots (multi-temperature)
# ------------------------
def plot_thermodynamics(all_snapshots, metadata=None):
    """
    Plots |Magnetization| and Energy vs Temperature in one figure.
    """
    if not all_snapshots or len(all_snapshots) <= 1:
        print("No multi-temperature data detected. Skipping thermodynamic plots.")
        return

    temps, avg_mags, avg_energies = [], [], []
    e_min = compute_min_energy(metadata or {})

    for temp, snaps in sorted(all_snapshots.items()):
        if not snaps: 
            continue
        mags = [abs(s["magnetization"]) for s in snaps]
        energies = [s["energy"] for s in snaps]

        coupling = snaps[0].get("J", 1.0)  # default to 1.0 if not present
        external_field = snaps[0].get("B", 0.0)  #

        temps.append(temp)
        avg_mags.append(np.mean(mags))
        avg_energy = np.mean(energies)
        if e_min is not None and e_min != 0:
            avg_energy = avg_energy / abs(e_min)
        avg_energies.append(avg_energy)

    # Create one figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))

    # |Magnetization| vs Temperature
    sns.lineplot(x=temps, y=avg_mags, marker="o", ax=axs[0])
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].set_xlabel("Temperature")
    axs[0].set_ylabel("|Magnetization|")
    axs[0].set_title("Average |Magnetization| vs Temperature for J={:.1f}, B={:.1f}".format(coupling, external_field))
    axs[0].grid(True)

    # Energy vs Temperature
    sns.lineplot(x=temps, y=avg_energies, marker="o", color="red", ax=axs[1])
    axs[1].set_xlabel("Temperature")
    energy_label = "Energy"
    if e_min is not None and e_min != 0:
        energy_label = "Energy / |E_min|"
    axs[1].set_ylim(-1.05, 0.05)
    axs[1].set_ylabel(energy_label)
    axs[1].set_title("Average Energy vs Temperature for J={:.1f}, B={:.1f}".format(coupling, external_field))
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# ------------------------
# Standalone test
# ------------------------
if __name__ == "__main__":
    file_path = resolve_snapshot_file()
    all_snapshots, metadata = load_snapshots(file_path)

    if len(all_snapshots) > 1:
        plot_thermodynamics(all_snapshots, metadata)
