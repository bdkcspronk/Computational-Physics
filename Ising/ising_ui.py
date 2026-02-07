import tkinter as tk
from tkinter import ttk
import threading
import os

import tkinter.filedialog as filedialog
import matplotlib.pyplot as plt
import ising_plots as plots
import ising_sim
import ising_viz

# ------------------------
# Globals
# ------------------------
stop_flag = False
simulation_done = False
progress_var = None
plot_energy_btn = None
plot_mag_btn = None
root = None

temp_widgets = []
temps = []
temp_bar_frame = None

directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

file_path = os.path.join("sim_data_ui", "snapshots.npz")

# ------------------------
# Utility functions 
# ------------------------

def compute_temperatures(params):
    if params.get("--multi_temp") != "ON":
        return [float(params["--start_T"])]

    Tmin = float(params["--min_T"])
    Tmax = float(params["--max_T"])
    n = int(params["--num_T"])

    if n <= 1:
        return [Tmin]

    return [Tmin + i*(Tmax - Tmin)/(n - 1) for i in range(n)]

def build_temp_bar(temps):
    for w in temp_bar_frame.winfo_children():
        w.destroy()

    temp_widgets.clear()

    for T in temps:
        lbl = tk.Label(
            temp_bar_frame,
            text=f"{T:.2f}",
            relief="ridge",
            padx=6,
            pady=3,
            bg="#dddddd"
        )
        lbl.pack(side="left", padx=2, fill="x", expand=True)
        temp_widgets.append(lbl)

def set_temp_state(i, state):
    if i < 0 or i >= len(temp_widgets):
        return

    if state == "queued":
        temp_widgets[i].config(bg="#dddddd")
    elif state == "running":
        temp_widgets[i].config(bg="#ffd966")
    elif state == "done":
        temp_widgets[i].config(bg="#93c47d")


def get_snapshot_keys(file_path):
    import numpy as np
    if not os.path.exists(file_path):
        return []

    data = np.load(file_path, allow_pickle=True)
    keys = []

    for key in data.files:
        # temperature keys are numeric
        try:
            float(key)
            keys.append(key)
        except ValueError:
            # skip non-numeric keys
            continue
    # fallback for single-temperature runs
    if not keys and 'snapshots' in data.files:
        keys.append('snapshots')

    return sorted(keys, key=lambda x: float(x))

# ------------------------
# Visualization function
# ------------------------
def visualize_snapshots(file_path, key):
    import tempfile, numpy as np, os
    data = np.load(file_path, allow_pickle=True)

    # Extract only the snapshots for this key
    if key == 'snapshots':
        snaps = data['snapshots']
    else:
        snaps = data[key]

    # Save temp .npz file
    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "snapshots.npz")
    save_dict = {'snapshots': snaps}
    # copy metadata
    for k in data.files:
        if k != key:
            save_dict[k] = data[k]
    np.savez_compressed(tmp_file, **save_dict)

    # Launch viz directly (no subprocess, better for executable builds)
    threading.Thread(
        target=ising_viz.run_visualizer,
        args=(tmp_dir,),
        daemon=True
    ).start()

def build_viz_buttons(file_path, parent_frame):
    # Clear only previous buttons, keep label(s)
    for widget in parent_frame.winfo_children():
        if isinstance(widget, ttk.Button):
            widget.destroy()

    keys = get_snapshot_keys(file_path)
    if not keys:
        ttk.Label(parent_frame, text="No snapshots found").grid(row=1, column=0)
        return
    
    for i, key in enumerate(keys):
        label = f"T={float(key):.2f}" if key != 'snapshots' else "Single Run"
        btn = ttk.Button(
            parent_frame,
                text=f"Visualize {label}",
            command=lambda k=key: visualize_snapshots(file_path, k)
        )
        btn.grid(row=i+1, column=0, sticky="ew", pady=2)

# ------------------------
# Simulation thread
# ------------------------
def run_simulation_thread(params):
    global stop_flag, simulation_done
    stop_flag = False
    simulation_done = False
    global cpu_workers
    cpu_workers = None

    output_dir = params["--output"]
    os.makedirs(output_dir, exist_ok=True)

    current_run = 0

    def handle_progress(event, value):
        nonlocal current_run
        global cpu_workers

        if event == "workers":
            cpu_workers = int(value)
            for i in range(min(cpu_workers, len(temp_widgets))):
                root.after(0, lambda i=i: set_temp_state(i, "running"))
            return

        if event == "finished_temp":
            finished = current_run
            current_run += 1

            def update():
                set_temp_state(finished, "done")

                next_idx = finished + (cpu_workers or 1)
                if next_idx < len(temp_widgets):
                    set_temp_state(next_idx, "running")

                run_var.set(f"Completed: {current_run}/{len(temp_widgets)}")

            root.after(0, update)

    ising_sim.run_from_params(params, progress_callback=handle_progress)

    def finish_ui():
        for i in range(len(temp_widgets)):
            set_temp_state(i, "done")

        plot_energy_btn.config(state="normal")
        plot_mag_btn.config(state="normal")
        plot_thermo_btn.config(state="normal")

    root.after(0, finish_ui)

    simulation_done = True

    root.after(0, lambda: build_viz_buttons(file_path, viz_buttons_frame))

    plot_energy_btn.config(state="normal")
    plot_mag_btn.config(state="normal")
    plot_thermo_btn.config(state="normal")

def start_simulation(entries):
    params = {
        "--H": entries["Lattice Height (H)"].get(),
        "--W": entries["Lattice Width (W)"].get(),
        "--L": entries["Lattice Depth (L)"].get(),
        "--start_T": entries["Start Temperature"].get(),
        "--target_T": entries["Target Temperature"].get(),
        "--dT": entries["ΔT per step"].get(),
        "--steps_dT": entries["Steps per ΔT"].get(),
        "--J": entries["Coupling J"].get(),
        "--B": entries["External Field B"].get(),
        "--steps": entries["Total MC Steps"].get(),
        "--relax_perc": entries["Relaxation (%)"].get(),
        "--stop_if_no_change": "ON" if entries["Stop if no change (ON/OFF)"].get() else "OFF",
        "--multi_temp": "ON" if entries["Enable Temperature Range (ON/OFF)"].get() else "OFF",
        "--min_T": entries["Min Temperature"].get(),
        "--max_T": entries["Max Temperature"].get(),
        "--num_T": entries["Number of Temperatures"].get(),
        "--dM_limit": entries["dM limit"].get(),
        "--nochange_X": entries["Number of snapshots X"].get(),
        "--stop_fully_mag": "ON" if entries["Stop if fully magnetized (ON/OFF)"].get() else "OFF",
        "--fully_mag_limit": entries["Fully magnetization limit"].get(),
        "--save": entries["Save Every N Steps"].get(),
        "--output": "sim_data_ui"
    }

    if entries["Random Seed (optional)"].get().strip():
        params["--seed"] = entries["Random Seed (optional)"].get().strip()

    global temps
    temps = compute_temperatures(params)

    build_temp_bar(temps)

    run_var.set(f"Completed: 0/{len(temps)}" if len(temps) > 1 else "Running...")

    plot_energy_btn.config(state="disabled")
    plot_mag_btn.config(state="disabled")
    plot_thermo_btn.config(state="disabled")

    threading.Thread(
        target=run_simulation_thread,
        args=(params,),
        daemon=True
    ).start()


def stop_simulation():
    global stop_flag
    stop_flag = True

def visualize_simulation():
    if simulation_done:
        threading.Thread(
            target=ising_viz.run_visualizer,
            args=("sim_data_ui",),
            daemon=True
        ).start()

# ------------------------
# Build GUI
# ------------------------

def add_group_frame(parent, title, explanation, settings):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(sticky='we', padx=5, pady=5, ipadx=5, ipady=5)
        frame.columnconfigure(1, weight=1)

        # Explanation text
        ttk.Label(frame, text=explanation, justify='left', wraplength=400).grid(
            row=0, column=0, columnspan=2, sticky='w', padx=5, pady=(0,5)
        )

        # Inputs
        for i, (label_text, default_val) in enumerate(settings, start=1):
            ttk.Label(frame, text=label_text).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            
            # Decide variable type for tickboxes or text
            if isinstance(default_val, str) and default_val.upper() in ("ON", "OFF"):
                var = tk.BooleanVar(value=(default_val.upper() == "ON"))
                entry_widget = ttk.Checkbutton(frame, variable=var)
                entry_widget.grid(row=i, column=1, sticky='e', padx=5, pady=2)
            else:
                var = tk.StringVar(value=str(default_val))
                entry_widget = ttk.Entry(frame, textvariable=var, width=15)
                entry_widget.grid(row=i, column=1, sticky='e', padx=5, pady=2)

            sim_entries[label_text] = var

        return frame

def build_gui():
    global progress_var, run_var, viz_buttons_frame, plot_energy_btn, plot_mag_btn, plot_thermo_btn, sim_entries, root
    root = tk.Tk()
    root.title("3D Ising Model Simulation")

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    sim_frame = ttk.Frame(notebook)
    notebook.add(sim_frame, text="Simulation Settings")

    sim_entries = {}
    row_idx = 0

    # Lattice Size
    lattice_settings = [
        ("Lattice Height (H)", 20),
        ("Lattice Width (W)", 20),
        ("Lattice Depth (L)", 20)
    ]
    add_group_frame(sim_frame, "Lattice Size", "Define the dimensions of the cubic lattice.", lattice_settings)
    row_idx += 1

    # MC Steps
    mc_settings = [
        ("Total MC Steps", 100),
        ("Save Every N Steps", 1),
    ]
    add_group_frame(
        sim_frame,
        "MC Steps",
        "Total MC steps specifies how many steps will be done for measurements.\n"
        "One step consists of H*W*L spin flip attempts.\n"
        "Save Every N Steps defines the interval at which snapshots are recorded.",        
        mc_settings
    )
    row_idx += 1

    # Relaxation steps
    relaxation_settings = [
        ("Relaxation (%)", 20)
    ]
    frame_relax = add_group_frame(
        sim_frame,
        "Relaxation Period",
        "Relaxation period is an initial fraction of steps (percentage of total MC steps) run before measurements begin.\n"
        "During relaxation, the system equilibrates and no snapshots are saved.",
        relaxation_settings
    )
    row_idx += 1

    # Add a dynamic label to show relaxation steps
    relax_label = ttk.Label(frame_relax, text="Relaxation steps: 0")
    relax_label.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=(2,5))

    def update_relax_label(*args):
        try:
            total_steps = int(sim_entries["Total MC Steps"].get())
            relax_perc = float(sim_entries["Relaxation (%)"].get())
            relax_steps = int(total_steps * relax_perc / 100)
            relax_label.config(text=f"Relaxation steps: {relax_steps}")
        except ValueError:
            relax_label.config(text="Relaxation steps: -")

    sim_entries["Total MC Steps"].trace_add("write", lambda *args: update_relax_label())
    sim_entries["Relaxation (%)"].trace_add("write", lambda *args: update_relax_label())
    update_relax_label()  # initialize

    # --- Temperature Settings ---
    temp_settings = [
        ("Start Temperature", 5.0),
        ("Target Temperature", 2.0),
        ("ΔT per step", 0.0),
        ("Steps per ΔT", 1)
    ]

    temp_explainer = (
        "Normal single-temperature run:\n"
        "- Start Temperature: initial temperature of the simulation.\n"
        "- Target Temperature: final temperature to reach.\n"
        "- ΔT per step: temperature change after every 'Steps per ΔT'.\n"
        "- Steps per ΔT: number of MC steps per temperature increment."
    )

    add_group_frame(
        sim_frame,
        "Temperature Settings",
        temp_explainer,
        temp_settings
    )
    row_idx += 1

    # Interaction Settings
    interaction_settings = [
        ("Coupling J", 1.0),
        ("External Field B", 0.0)
    ]
    add_group_frame(sim_frame, "Interaction Settings", "Define spin-spin coupling and external magnetic field.", interaction_settings)
    row_idx += 1

    # --- New frame below notebook ---
    bottom_frame = ttk.Frame(root)
    bottom_frame.pack(fill='x', padx=10, pady=5)

    global temp_bar_frame
    temp_bar_frame = ttk.Frame(bottom_frame)
    temp_bar_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))

    run_var = tk.StringVar(value="Run: -")
    ttk.Label(bottom_frame, textvariable=run_var).pack(side="left", padx=(0,10))

    ttk.Button(bottom_frame, text="Run Simulation", command=lambda: start_simulation(sim_entries)).pack(side='left', padx=5)
    ttk.Button(bottom_frame, text="Stop Simulation", command=stop_simulation).pack(side='left', padx=5)


    # --- Tab 2: Additional settings ---
    early_stop_tab = ttk.Frame(root)  # just a Frame, not a Notebook inside a Notebook
    notebook.add(early_stop_tab, text="Advanced Settings")

     # --- Temperature Range for Multi-Temperature Runs ---
    temp_range_settings = [
        ("Enable Temperature Range (ON/OFF)", "OFF"),
        ("Min Temperature", 2.0),
        ("Max Temperature", 5.0),
        ("Number of Temperatures", 4)
    ]

    temp_range_explainer = (
        "Optional multi-temperature run:\n"
        "- Enable to run simulations at multiple temperatures.\n"
        "- This overrides the single start/target temperature if enabled.\n"
        "- Number of Temperatures specifies how many temperatures to sample between Min and Max Temperature.\n"
        "- NOTE: ΔT and ΔT steps settings still apply within each temperature."
    )

    add_group_frame(
        early_stop_tab,
        "Multi-Temperature Settings",
        temp_range_explainer,
        temp_range_settings
    )

    # Steady state stop group
    ss_settings = [
        ("Stop if no change (ON/OFF)", "OFF"),
        ("dM limit", 0.001),
        ("Number of snapshots X", 5)
    ]

    ss_explainer = (
        "Stop the simulation early if the system reaches a steady state (i.e., no significant change in magnetization over recent snapshots).\n"
        "- 'Stop if no change' enables/disables this feature.\n"
        "- 'dM limit' is the threshold for average change in magnetization.\n"
        "- 'Number of snapshots X' specifies how many recent snapshots to average over."
    )

    add_group_frame(
        early_stop_tab,
        "Steady State Stop",
        ss_explainer,
        ss_settings
    )

    # Fully magnetized stop group
    fully_mag_settings = [
        ("Stop if fully magnetized (ON/OFF)", "OFF"),
        ("Fully magnetization limit", .99)
    ]

    fully_mag_explainer = (
        "Stop the simulation early if the system reaches near full magnetization.\n"
        "- 'Stop if fully magnetized' enables this feature.\n"
        "- 'Fully magnetization limit' sets the threshold to trigger stopping."
    )

    add_group_frame(
        early_stop_tab,
        "Fully Magnetized Stop",
        fully_mag_explainer,
        fully_mag_settings
    )

    # Seed Settings
    seed_settings = [
        ("Random Seed (optional)", "")
    ]
    add_group_frame(early_stop_tab, "Random Seed", "Set a random seed for reproducibility.", seed_settings)

    # --- Tab 3: Plots and animation ---
    plots_tab = ttk.Notebook(root)
    notebook.add(plots_tab, text="Analysis")

    # Frame inside the tab to hold plot buttons
    plot_buttons_frame = ttk.Frame(plots_tab)
    plot_buttons_frame.pack(fill="both", expand=True, padx=10, pady=10)
    plot_buttons_frame.columnconfigure(0, weight=1)

    # Buttons for Energy / Magnetization plots (static)
    plot_energy_btn = ttk.Button(
        plot_buttons_frame,
        text="Plot Energy vs Step",
        command=lambda: plots.plot_energy_time(file_path=file_path),
        state="disabled"
    )
    plot_mag_btn = ttk.Button(
        plot_buttons_frame,
        text="Plot Magnetization vs Step",
        command=lambda: plots.plot_mag_time(file_path=file_path),
        state="disabled"
    )

    plot_thermo_btn = ttk.Button(
        plot_buttons_frame,
        text="Plot Thermodynamics",
        command=lambda: plots.plot_thermodynamics(plots.load_snapshots(file_path)[0]),
        state="disabled"
    )

    plot_energy_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    plot_mag_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    plot_thermo_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

    # --- Dynamic Visualization Buttons ---
    viz_container = ttk.LabelFrame(plot_buttons_frame, text="Visualize Snapshots")
    viz_container.grid(row=3, column=0, padx=5, pady=10, sticky="nsew")
    viz_container.columnconfigure(0, weight=1)

    # Keyboard instructions (always visible)
    keyboard_label = ttk.Label(
        viz_container,
        text="Keyboard during animation:\n"
            "R: reverse play\n"
            "SPACE: pause/play\n"
            "UP/DOWN: adjust speed\n"
            "ESC: close",
        justify="left",
        wraplength=300
    )
    keyboard_label.grid(row=0, column=0, padx=5, pady=(5,5), sticky="w")

    # Canvas for scrollable buttons
    canvas = tk.Canvas(viz_container, height=200)  # fixed height
    canvas.grid(row=1, column=0, sticky="nsew")

    # Scrollbar
    scrollbar = ttk.Scrollbar(viz_container, orient="vertical", command=canvas.yview)
    scrollbar.grid(row=1, column=1, sticky="ns")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Frame inside canvas to hold buttons
    viz_buttons_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=viz_buttons_frame, anchor="nw")

    # Make the canvas scrollable when the frame size changes
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    viz_buttons_frame.bind("<Configure>", on_frame_configure)

    # Initial population of snapshot buttons
    build_viz_buttons(file_path, viz_buttons_frame)

    # --- Tab 4: Ising Model Info ---
    info_frame = ttk.Frame(notebook)
    notebook.add(info_frame, text="Ising Model Info")
    info_text = (
        "3D Ising Model Simulation\n\n"
        "The system is a cubic lattice of spins (+1 or -1), with energy determined by\n"
        "nearest-neighbor interactions and an optional external magnetic field.\n\n"
        "Simulation uses the Metropolis Monte Carlo algorithm:\n"
        " - One Monte Carlo step (MCS) consists of N = H × W × L single-spin flip attempts\n"
        " - For each attempt:\n"
        "     * A spin is chosen at random\n"
        "     * The energy change ΔE if it were flipped is computed\n"
        "     * The spin flips with probability min(1, exp(-ΔE / T))\n\n"
        "Temperature can change during the simulation according to ΔT per step.\n"
        "Snapshots of the lattice are saved every N steps, excluding the initial relaxation period.\n"
    )

    ttk.Label(info_frame, text=info_text, justify='left', wraplength=600).pack(padx=10, pady=10, anchor='w')

    root.mainloop()

# ------------------------
# Run GUI
# ------------------------
if __name__ == "__main__":
    build_gui()
