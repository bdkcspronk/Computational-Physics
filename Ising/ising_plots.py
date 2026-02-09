import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
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

    # Case 1: single-temperature run saved as 'snapshots'
    if 'snapshots' in data.files:
        snapshots = data['snapshots'].tolist()
        params = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files if k != 'snapshots'}
        temp_key = float(params['target_T']) if 'target_T' in params else 1.0
        return {temp_key: snapshots}, params

    # Case 2: multi-temperature run saved as temperature keys
    all_snapshots = {}
    for key in data.files:
        try:
            temp = float(key)
            all_snapshots[temp] = data[key].tolist()
        except ValueError:
            continue  # skip non-temperature keys
    return all_snapshots, {}


# ------------------------
# Time evolution plots
# ------------------------

def plot_energy_time(file_path):
    all_snapshots, _ = load_snapshots(file_path)
    if not all_snapshots:
        print("No snapshots found.")
        return

    temps_sorted = sorted(all_snapshots.keys())
    cmap = cm.get_cmap("viridis", len(temps_sorted))  # gradient
    plt.figure(figsize=(8,5))

    for i, temp in enumerate(temps_sorted):
        snaps = all_snapshots[temp]
        if not snaps:
            continue
        steps = [s["step"] for s in snaps]
        energy = [s["energy"] for s in snaps]
        coupling = snaps[0].get("J", 1.0)  # default to 1.0 if not present
        external_field = snaps[0].get("B", 0.0)  # default to 0.0 if not present

        plt.plot(steps, energy, color=cmap(i), label=f"T={temp:.3g}")

    plt.xlabel(f"MC Steps for J={coupling}, B={external_field}")
    plt.ylabel("Energy")
    plt.title("Energy vs Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mag_time(file_path):
    all_snapshots, _ = load_snapshots(file_path)
    if not all_snapshots:
        print("No snapshots found.")
        return

    temps_sorted = sorted(all_snapshots.keys())
    cmap = cm.get_cmap("viridis", len(temps_sorted))
    plt.figure(figsize=(8,5))

    for i, temp in enumerate(temps_sorted):
        snaps = all_snapshots[temp]
        if not snaps:
            continue
        steps = [s["step"] for s in snaps]
        mag = [s["magnetization"] for s in snaps]
        coupling = snaps[0].get("J", 1.0)  # default to 1.0 if not present
        external_field = snaps[0].get("B", 0.0)  #

        plt.plot(steps, mag, color=cmap(i), label=f"T={temp:.3g}")

    plt.ylim(-1.05, 1.05)
    plt.xlabel(f"MC Steps for J={coupling}, B={external_field}")
    plt.ylabel("Magnetization")
    plt.title("Magnetization vs Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------
# Thermodynamic plots (multi-temperature)
# ------------------------
def plot_thermodynamics(all_snapshots):
    """
    Plots |Magnetization| and Energy vs Temperature in one figure.
    """
    if not all_snapshots or len(all_snapshots) <= 1:
        print("No multi-temperature data detected. Skipping thermodynamic plots.")
        return

    temps, avg_mags, avg_energies = [], [], []

    for temp, snaps in sorted(all_snapshots.items()):
        if not snaps: 
            continue
        mags = [abs(s["magnetization"]) for s in snaps]
        energies = [s["energy"] for s in snaps]

        coupling = snaps[0].get("J", 1.0)  # default to 1.0 if not present
        external_field = snaps[0].get("B", 0.0)  #

        temps.append(temp)
        avg_mags.append(np.mean(mags))
        avg_energies.append(np.mean(energies))

    # Create one figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))

    # |Magnetization| vs Temperature
    sns.lineplot(x=temps, y=avg_mags, marker="o", ax=axs[0])
    axs[0].set_xlabel("Temperature")
    axs[0].set_ylabel("|Magnetization|")
    axs[0].set_title("Average |Magnetization| vs Temperature for J={:.1f}, B={:.1f}".format(coupling, external_field))
    axs[0].grid(True)

    # Energy vs Temperature
    sns.lineplot(x=temps, y=avg_energies, marker="o", color="red", ax=axs[1])
    axs[1].set_xlabel("Temperature")
    axs[1].set_ylabel("Energy")
    axs[1].set_title("Average Energy vs Temperature for J={:.1f}, B={:.1f}".format(coupling, external_field))
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# ------------------------
# Standalone test
# ------------------------
if __name__ == "__main__":
    file_path = resolve_snapshot_file()
    all_snapshots, _ = load_snapshots(file_path)

    if len(all_snapshots) > 1:
        plot_thermodynamics(all_snapshots)