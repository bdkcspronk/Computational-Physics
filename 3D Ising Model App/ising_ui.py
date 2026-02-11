from ttkthemes import ThemedTk
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import multiprocessing
import threading
import os
import sys
import ctypes
import json
import re
from numpy import load
import ising_plots as plots
from ising_initialization import ensure_open_dir, resolve_open_dir, resolve_snapshot_file
import ising_sim
import ising_viz

# ------------------------
# Globals
# ------------------------
stop_flag = False
simulation_done = False
sim_stop_event = None
progress_var = None
plot_energy_btn = None
plot_mag_btn = None
root = None
thermo_status_var = None
ANALYSIS_BTN_WIDTH = 24

viz_process = None
temp_widgets = []
temps = []
temp_bar_frame = None
run_index_entries = []
runs_status_var = None
runs_tree = None
runs_tree_records = {}
runs_tree_anchor = None
checkbox_images = None

# Global UI colors
UI_COLORS = {
    "fg": "#E0E0E0",              # light text
    "text": "#FFFFFF",             # main text
    "muted": "#A0A0A0",            # secondary text
    "panel": "#1E1E2F",            # background panels
    "frame": "#2A2A3F",            # frames around panels
    "labelframe": "#262635",       
    "labelframe_label": "#D0D0FF", # label text
    "label": "#FFD080",            # UI labels
    "label_bg": "#2A2A3F",         
    "frame_label_bg": "#3A3A50",   
    "tab": "#1E1E2F",              
    "tab_selected": "#FFD080",     
    "tree_bg": "#1A1A2A",          
    "tree_header": "#2A2A3F",      
    "entry_bg": "#2A2A3F",         
    "progress_trough": "#262635",  
    "analysis_bg": "#FFD080",      
    "border": "#FFD080",           
    "border_light": "#A0A0FF",     
    "accent": "#7F5FFF",           # main accent color
    "accent_dark": "#5F3FFF",      
    "header": "#FFB86C",           
    "temp_queued": "#888888",      
    "temp_running": "#FFD700",     
    "temp_done": "#50FA7B",        
}

UI_PADDING = {
    "outer": 10,
    "inner": 10,
    "section": 8,
    "label": 5,
    "tight": 2,
    "wide": 10,
    "frame_label": 8,
}

file_path = resolve_snapshot_file()
active_snapshot_file = file_path

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
            bg=UI_COLORS["temp_queued"],
            fg=UI_COLORS["text"]
        )
        lbl.pack(side="left", padx=2, fill="x", expand=True)
        temp_widgets.append(lbl)

def set_temp_state(i, state):
    if i < 0 or i >= len(temp_widgets):
        return

    if state == "queued":
        temp_widgets[i].config(bg=UI_COLORS["temp_queued"])
    elif state == "running":
        temp_widgets[i].config(bg=UI_COLORS["temp_running"])
    elif state == "done":
        temp_widgets[i].config(bg=UI_COLORS["temp_done"])


def _extract_temp_from_key(key):
    try:
        return float(key)
    except (TypeError, ValueError):
        pass
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(key))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None

def get_snapshot_keys(file_path):
    if not os.path.exists(file_path):
        return []

    with load(file_path, allow_pickle=True) as data:
        numeric_keys = []

        for key in data.files:
            # temperature keys are numeric
            temp_val = _extract_temp_from_key(key)
            if temp_val is None:
                continue
            numeric_keys.append((temp_val, key))

        if numeric_keys:
            return [key for _, key in sorted(numeric_keys, key=lambda item: item[0])]

        # fallback for single-temperature runs
        if 'snapshots' in data.files:
            return ['snapshots']

    return []

def get_active_snapshot_file():
    return active_snapshot_file

def set_active_snapshot_file(path):
    global active_snapshot_file
    active_snapshot_file = path

def _build_checkbox_images(root, size=24):
    def draw_line(img, x0, y0, x1, y1, color, thickness=2):
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        for s in range(steps + 1):
            x = int(round(x0 + dx * s / steps))
            y = int(round(y0 + dy * s / steps))
            for tx in range(-(thickness // 2), thickness // 2 + 1):
                for ty in range(-(thickness // 2), thickness // 2 + 1):
                    px = x + tx
                    py = y + ty
                    if 0 <= px < size and 0 <= py < size:
                        img.put(color, (px, py))

    off = tk.PhotoImage(master=root, width=size, height=size)
    on = tk.PhotoImage(master=root, width=size, height=size)

    bg = UI_COLORS["panel"]
    border = UI_COLORS["border"]
    border_light = UI_COLORS["border_light"]
    accent = UI_COLORS["accent"]
    fg = UI_COLORS["text"]

    off.put(bg, to=(0, 0, size, size))
    on.put(bg, to=(0, 0, size, size))

    for i in range(size):
        off.put(border_light, (i, 0))
        off.put(border_light, (i, size - 1))
        off.put(border_light, (0, i))
        off.put(border_light, (size - 1, i))

        on.put(border, (i, 0))
        on.put(border, (i, size - 1))
        on.put(border, (0, i))
        on.put(border, (size - 1, i))

    inset = 2
    on.put(accent, to=(inset, inset, size - inset, size - inset))
    draw_line(on, 6, 13, 10, 17, fg, thickness=2)
    draw_line(on, 10, 17, 18, 7, fg, thickness=2)

    return {"off": off, "on": on}

def create_scrollable_tab(notebook, title):
    container = ttk.Frame(notebook, style="Tab.TFrame")
    notebook.add(container, text=title)
    container.columnconfigure(0, weight=1)
    container.rowconfigure(0, weight=1)

    canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0, bg=UI_COLORS["analysis_bg"])
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    content = ttk.Frame(canvas, style="Tab.TFrame")
    window_id = canvas.create_window((0, 0), window=content, anchor="nw")

    def update_scrollbar():
        bbox = canvas.bbox("all")
        if not bbox:
            scrollbar.grid_remove()
            return
        content_height = bbox[3] - bbox[1]
        if content_height <= canvas.winfo_height():
            scrollbar.grid_remove()
        else:
            scrollbar.grid()

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        update_scrollbar()

    def on_canvas_configure(event):
        canvas.itemconfig(window_id, width=event.width)
        update_scrollbar()

    content.bind("<Configure>", on_frame_configure)
    canvas.bind("<Configure>", on_canvas_configure)

    def on_mousewheel(event):
        target = canvas.winfo_containing(event.x_root, event.y_root)
        if target is None or not str(target).startswith(str(container)):
            return
        bbox = canvas.bbox("all")
        if not bbox:
            return
        content_height = bbox[3] - bbox[1]
        if content_height <= canvas.winfo_height():
            return
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel, add="+")

    return content

def can_plot_thermodynamics(file_path):
    if not os.path.exists(file_path):
        return False
    try:
        all_snapshots, _ = plots.load_snapshots(file_path)
    except Exception:
        return False
    return len(all_snapshots) > 1

def update_thermo_button_state():
    if plot_thermo_btn is None:
        return
    if can_plot_thermodynamics(get_active_snapshot_file()):
        plot_thermo_btn.config(state="normal")
        if thermo_status_var is not None:
            thermo_status_var.set("")
    else:
        plot_thermo_btn.config(state="disabled")
        if thermo_status_var is not None:
            thermo_status_var.set("Thermodynamics requires multiple temperatures.")

def load_saved_runs_index():
    open_dir = resolve_open_dir()
    index_filename = getattr(ising_sim, "_INDEX_FILENAME", "simulation_index.jsonl")
    index_path = os.path.join(open_dir, index_filename)
    if not os.path.exists(index_path):
        return []

    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache_path = rec.get("cache_path")
            if cache_path and not os.path.isabs(cache_path):
                cache_path = os.path.join(open_dir, cache_path)
                rec["cache_path"] = cache_path
            entries.append(rec)
    return entries

def _clean_sig_value(value):
    value = value.strip().strip("'\"")
    match = re.match(r"np\.[^(]+\((.*)\)", value)
    if match:
        value = match.group(1)
    return value

def _parse_signature(signature):
    data = {}
    if not signature:
        return data
    for part in signature.split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[key] = _clean_sig_value(value)
    return data

def _format_run_label(rec):
    if rec.get("description"):
        return rec["description"]

    sig_data = _parse_signature(rec.get("signature", ""))
    temp = rec.get("temp")
    if temp is None:
        temp = sig_data.get("start_T", "?")
    h = sig_data.get("H", "?")
    w = sig_data.get("W", "?")
    l = sig_data.get("L", "?")
    j = sig_data.get("J", "?")
    b = sig_data.get("B", "?")
    steps = sig_data.get("steps", "?")
    seed = sig_data.get("seed", "None")

    return f"T={temp}, H={h}, W={w}, L={l}, J={j}, B={b}, steps={steps}, seed={seed}"

def _tree_sort_key(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)

def _sort_treeview(tree, col, reverse):
    items = [(tree.set(k, col), k) for k in tree.get_children("")]
    items.sort(key=lambda pair: _tree_sort_key(pair[0]), reverse=reverse)
    for index, (_, iid) in enumerate(items):
        tree.move(iid, "", index)
    tree.heading(col, command=lambda: _sort_treeview(tree, col, not reverse))

def populate_saved_runs():
    global run_index_entries, runs_tree_records

    if runs_tree is None:
        return

    runs_tree.delete(*runs_tree.get_children())
    runs_tree_records = {}

    run_index_entries = [
        rec for rec in load_saved_runs_index()
        if rec.get("type") == "temp" and rec.get("cache_path")
    ]

    if not run_index_entries:
        if runs_status_var is not None:
            runs_status_var.set("No saved runs found.")
        return

    for rec in run_index_entries:
        sig_data = _parse_signature(rec.get("signature", ""))
        temp = rec.get("temp")
        if temp is None:
            temp = sig_data.get("start_T", "")
        values = (
            temp,
            sig_data.get("H", ""),
            sig_data.get("W", ""),
            sig_data.get("L", ""),
            sig_data.get("J", ""),
            sig_data.get("B", ""),
            sig_data.get("dT", ""),
            sig_data.get("steps_dT", ""),
            sig_data.get("steps", ""),
            sig_data.get("seed", ""),
        )
        iid = rec.get("id") or str(len(runs_tree_records))
        runs_tree.insert("", "end", iid=iid, values=values)
        runs_tree_records[iid] = rec

    if runs_status_var is not None:
        runs_status_var.set(f"Loaded {len(run_index_entries)} runs.")

def select_all_runs():
    if runs_tree is None:
        return
    runs_tree.selection_set(runs_tree.get_children(""))

def clear_run_selection():
    if runs_tree is None:
        return
    runs_tree.selection_remove(runs_tree.selection())

def _tree_update_anchor():
    global runs_tree_anchor
    if runs_tree is None:
        return
    focus = runs_tree.focus()
    if focus:
        runs_tree_anchor = focus

def _tree_shift_select(direction):
    global runs_tree_anchor
    if runs_tree is None:
        return "break"
    items = list(runs_tree.get_children(""))
    if not items:
        return "break"

    focus = runs_tree.focus() or items[0]
    if focus not in items:
        focus = items[0]

    try:
        idx = items.index(focus)
    except ValueError:
        idx = 0

    new_idx = max(0, min(len(items) - 1, idx + direction))
    target = items[new_idx]

    if runs_tree_anchor is None or runs_tree_anchor not in items:
        runs_tree_anchor = focus

    anchor_idx = items.index(runs_tree_anchor)
    start = min(anchor_idx, new_idx)
    end = max(anchor_idx, new_idx)

    runs_tree.selection_set(items[start:end + 1])
    runs_tree.focus(target)
    runs_tree.see(target)
    return "break"

def delete_selected_runs():
    global runs_status_var

    selected = []
    if runs_tree is not None:
        for iid in runs_tree.selection():
            rec = runs_tree_records.get(iid)
            if rec:
                selected.append(rec)
    if not selected:
        if runs_status_var is not None:
            runs_status_var.set("No runs selected to delete.")
        return

    delete_ids = {rec.get("id") for rec in selected if rec.get("id")}
    deleted = []
    failed = []

    for rec in selected:
        cache_path = rec.get("cache_path")
        if cache_path and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                deleted.append(rec.get("id", "unknown"))
            except OSError:
                failed.append(rec.get("id", "unknown"))
        else:
            failed.append(rec.get("id", "unknown"))

    open_dir = resolve_open_dir()
    index_filename = getattr(ising_sim, "_INDEX_FILENAME", "simulation_index.jsonl")
    index_path = os.path.join(open_dir, index_filename)
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(index_path, "w", encoding="utf-8") as f:
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("id") in delete_ids:
                        continue
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        except OSError:
            pass

    populate_saved_runs()
    if runs_status_var is not None:
        status = f"Deleted {len(deleted)}/{len(selected)} runs."
        if failed:
            status += f" Failed: {', '.join(failed)}"
        runs_status_var.set(status)

def combine_selected_runs():
    global runs_status_var

    selected = []
    if runs_tree is not None:
        for iid in runs_tree.selection():
            rec = runs_tree_records.get(iid)
            if rec:
                selected.append(rec)
    if not selected:
        if runs_status_var is not None:
            runs_status_var.set("No runs selected.")
        return

    combined = {}
    base_meta = None
    base_dims = None
    skipped = []

    for rec in selected:
        cache_path = rec.get("cache_path")
        if not cache_path or not os.path.exists(cache_path):
            skipped.append(rec.get("id", "unknown"))
            continue
        try:
            cached = ising_sim.load_cached_run(cache_path)
        except Exception:
            skipped.append(rec.get("id", "unknown"))
            continue

        snapshots_by_temp = cached.get("snapshots_by_temp", {})
        metadata = cached.get("metadata", {}) or {}
        dims = (metadata.get("H"), metadata.get("W"), metadata.get("L"))

        if base_dims is None:
            base_dims = dims
            base_meta = metadata
        else:
            if all(d is not None for d in base_dims) and all(d is not None for d in dims):
                if dims != base_dims:
                    skipped.append(rec.get("id", "unknown"))
                    continue

        if not metadata and base_meta:
            metadata = base_meta

        for temp_key, snaps in snapshots_by_temp.items():
            temp_val = _extract_temp_from_key(temp_key)
            if temp_val is None:
                continue
            base_key = str(temp_val)
            snaps_list = list(snaps)
            if base_key not in combined:
                combined[base_key] = snaps_list
            else:
                rec_id = rec.get("id", "run")
                unique_key = f"T={temp_val}_run={rec_id}"
                suffix = 1
                while unique_key in combined:
                    suffix += 1
                    unique_key = f"T={temp_val}_run={rec_id}_{suffix}"
                combined[unique_key] = snaps_list

    if not combined:
        if runs_status_var is not None:
            runs_status_var.set("No compatible runs selected.")
        return

    output_path = os.path.join(resolve_open_dir(), "snapshots_selected.npz")
    ising_sim.save_combined_npz(output_path, combined, base_meta or {})
    set_active_snapshot_file(output_path)
    build_viz_buttons(get_active_snapshot_file(), viz_buttons_frame)
    plot_energy_btn.config(state="normal")
    plot_mag_btn.config(state="normal")
    update_thermo_button_state()

    if runs_status_var is not None:
        status = f"Loaded {len(selected) - len(skipped)}/{len(selected)} runs into snapshots_selected.npz."
        if skipped:
            status += f" Skipped: {', '.join(skipped)}"
        runs_status_var.set(status)

# ------------------------
# Visualization function
# ------------------------
def visualize_snapshots(file_path):
    global viz_process

    if not os.path.exists(file_path):
        return

    # Keep a single visualizer process alive at a time to avoid pygame/OpenGL crashes
    if viz_process is not None and viz_process.is_alive():
        viz_process.terminate()
        viz_process.join(timeout=2)

    viz_process = multiprocessing.Process(
        target=ising_viz.run_visualizer,
        args=(file_path,),
        daemon=True
    )
    viz_process.start()

def build_viz_buttons(file_path, parent_frame):
    # Clear previous buttons/labels inside the scrollable frame
    for widget in parent_frame.winfo_children():
        widget.destroy()

    keys = get_snapshot_keys(file_path)
    if not keys:
        ttk.Label(parent_frame, text="No snapshots found").grid(row=0, column=0)
        return

    btn = ttk.Button(
        parent_frame,
        text="Visualize Snapshots",
        command=lambda: visualize_snapshots(get_active_snapshot_file()),
        width=ANALYSIS_BTN_WIDTH
    )
    btn.grid(row=0, column=0, sticky="ew", pady=2)

# ------------------------
# Simulation thread
# ------------------------
def run_simulation_thread(params):
    global stop_flag, simulation_done, sim_stop_event
    stop_flag = False
    simulation_done = False
    sim_stop_event = threading.Event()
    global cpu_workers
    cpu_workers = None

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

    ising_sim.run_from_params(
        params,
        progress_callback=handle_progress,
        stop_requested=lambda: stop_flag or (sim_stop_event is not None and sim_stop_event.is_set())
    )

    def finish_ui():
        for i in range(len(temp_widgets)):
            set_temp_state(i, "done")

        plot_energy_btn.config(state="normal")
        plot_mag_btn.config(state="normal")
        set_active_snapshot_file(resolve_snapshot_file())
        update_thermo_button_state()
        populate_saved_runs()

    root.after(0, finish_ui)

    simulation_done = True

    root.after(0, lambda: build_viz_buttons(get_active_snapshot_file(), viz_buttons_frame))


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
        "--output": ensure_open_dir()
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
    update_thermo_button_state()

    threading.Thread(
        target=run_simulation_thread,
        args=(params,),
        daemon=True
    ).start()


def stop_simulation():
    global stop_flag, sim_stop_event
    stop_flag = True
    if sim_stop_event is not None:
        sim_stop_event.set()
    ising_sim.request_stop()
    run_var.set("Stopping simulation...")

def visualize_simulation():
    if simulation_done:
        threading.Thread(
            target=ising_viz.run_visualizer,
            args=(get_active_snapshot_file(),),
            daemon=True
        ).start()

# ------------------------
# Build GUI
# ------------------------

def add_group_frame(parent, title, explanation, settings):
        frame = ttk.LabelFrame(parent)
        label_widget = ttk.Label(
            frame,
            text=title,
            style="FrameLabel.TLabel",
            padding=(UI_PADDING["frame_label"], 0, UI_PADDING["frame_label"], 0),
        )
        frame.configure(labelwidget=label_widget)
        frame.grid(
            sticky='we',
            padx=UI_PADDING["section"],
            pady=UI_PADDING["section"],
            ipadx=UI_PADDING["inner"],
            ipady=UI_PADDING["inner"],
        )
        frame.columnconfigure(1, weight=1)

        # Explanation text
        ttk.Label(frame, text=explanation, justify='left', wraplength=400).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky='w',
            padx=UI_PADDING["label"],
            pady=(0, UI_PADDING["label"]),
        )

        # Inputs
        grid_row = 1
        for label_text, default_val in settings:
            
            # Decide variable type for tickboxes or text
            if isinstance(default_val, str) and default_val.upper() in ("ON", "OFF"):
                ttk.Label(frame, text=label_text).grid(
                    row=grid_row,
                    column=1,
                    sticky='e',
                    padx=UI_PADDING["label"],
                    pady=(UI_PADDING["tight"], 0),
                )
                grid_row += 1
                var = tk.BooleanVar(value=(default_val.upper() == "ON"))
                entry_widget = tk.Checkbutton(
                    frame,
                    variable=var,
                    bg=UI_COLORS["header"],
                    fg=UI_COLORS["text"],
                    activebackground=UI_COLORS["header"],
                    activeforeground=UI_COLORS["text"],
                    selectcolor=UI_COLORS["panel"],
                    highlightthickness=0,
                    borderwidth=0,
                    image=checkbox_images["off"] if checkbox_images else None,
                    selectimage=checkbox_images["on"] if checkbox_images else None,
                    indicatoron=False,
                    width=24,
                    height=24,
                    font=("Bahnschrift", 16),
                )
                entry_widget.grid(
                    row=grid_row,
                    column=1,
                    sticky='e',
                    padx=UI_PADDING["label"],
                    pady=(0, UI_PADDING["inner"]),
                )
                grid_row += 1
            else:
                ttk.Label(frame, text=label_text).grid(
                    row=grid_row,
                    column=0,
                    sticky='w',
                    padx=UI_PADDING["label"],
                    pady=UI_PADDING["tight"],
                )
                var = tk.StringVar(value=str(default_val))
                entry_widget = ttk.Entry(frame, textvariable=var, width=15)
                entry_widget.grid(
                    row=grid_row,
                    column=1,
                    sticky='e',
                    padx=UI_PADDING["label"],
                    pady=UI_PADDING["tight"],
                )
                grid_row += 1

            sim_entries[label_text] = var

        return frame

def build_gui():
    global progress_var, run_var, viz_buttons_frame, plot_energy_btn, plot_mag_btn, plot_thermo_btn, sim_entries, root
    global runs_status_var, runs_tree, thermo_status_var, checkbox_images
    is_windows = sys.platform.startswith("win")
    if ThemedTk is not None and not is_windows:
        root = ThemedTk(theme="yaru")
    else:
        root = tk.Tk()
    root.title("3D Ising Model Simulation")

    style = ttk.Style(root)
    if is_windows or ThemedTk is None:
        # Prefer native Windows themes for performance.
        for theme in ("vista", "xpnative", "clam"):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue

    # --- Minimal dark theme pass (native theme + light styling) ---
    base_font = ("Bahnschrift", 10)
    root.option_add("*Font", base_font)
    style.configure(".", font=base_font)

    colors = UI_COLORS

    if is_windows:
        # Clam respects custom colors for notebook tabs and frames on Windows.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    root.configure(bg=colors["panel"])
    checkbox_images = _build_checkbox_images(root, size=24)
    style.configure("TFrame", background=colors["frame"])
    style.configure("Tab.TFrame", background=colors["analysis_bg"])
    style.configure("Analysis.TFrame", background=colors["analysis_bg"])
    style.configure("TLabelframe", background=colors["labelframe"], padding=(12, 10), borderwidth=4, relief="groove")
    style.configure("TLabelframe.Label", background=colors["frame_label_bg"], foreground=colors["text"], font=("Bahnschrift", 10, "bold"))
    style.configure("FrameLabel.TLabel", background=colors["frame_label_bg"], foreground=colors["text"], font=("Bahnschrift", 10, "bold"))
    style.configure("TLabel", background=colors["label_bg"], foreground=colors["text"])

    style.configure("TButton", padding=(8, 4))
    style.map(
        "TButton",
        foreground=[("disabled", colors["muted"])],
        background=[("active", colors["accent_dark"])],
    )

    style.configure("TCheckbutton", padding=(6, 4), background=colors["labelframe"], foreground=colors["text"])
    style.map(
        "TCheckbutton",
        background=[("active", colors["labelframe"])],
        foreground=[("disabled", colors["muted"])],
        indicatorcolor=[("selected", colors["accent"]), ("!selected", colors["labelframe"])],
    )

    style.configure("TNotebook", background=colors["panel"], borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        padding=(12, 6),
        background=colors["tab"],
        foreground=colors["text"],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["tab_selected"])],
        foreground=[("selected", colors["text"])],
    )

    style.configure(
        "Treeview",
        background=colors["tree_bg"],
        fieldbackground=colors["tree_bg"],
        foreground=colors["text"],
        rowheight=22,
        bordercolor=colors["border"],
        lightcolor=colors["border"],
        darkcolor=colors["border"],
    )
    style.map("Treeview", background=[("selected", colors["accent"])])
    style.configure(
        "Treeview.Heading",
        background=colors["tree_header"],
        foreground=colors["text"],
        font=("Bahnschrift", 10, "bold"),
    )

    # Tk widgets that don't inherit ttk styles
    root.option_add("*Canvas.Background", colors["panel"])
    root.option_add("*Label.Background", colors["label_bg"])
    root.option_add("*Label.Foreground", colors["text"])
    style.configure("TEntry", fieldbackground=colors["entry_bg"], foreground=colors["text"])
    style.map("TEntry", fieldbackground=[("disabled", colors["entry_bg"])])
    style.configure("TProgressbar", background=colors["accent"], troughcolor=colors["progress_trough"], bordercolor=colors["border"])

    def enable_dark_titlebar(window):
        if not is_windows:
            return
        try:
            hwnd = window.winfo_id()
            value = ctypes.c_int(1)
            # Try newer (20) then older (19) immersive dark mode attribute.
            for attr in (20, 19):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attr,
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                )
        except Exception:
            pass

    enable_dark_titlebar(root)
    root.geometry("900x1100")  # pick a size you like
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=UI_PADDING["outer"], pady=UI_PADDING["outer"])

    sim_frame = create_scrollable_tab(notebook, "Simulation Settings")
    sim_frame.columnconfigure(0, weight=1)

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
    bottom_frame.pack(fill='x', padx=UI_PADDING["wide"], pady=UI_PADDING["outer"])

    global temp_bar_frame
    temp_bar_frame = ttk.Frame(bottom_frame)
    temp_bar_frame.pack(side="left", fill="x", expand=True, padx=(0, UI_PADDING["wide"]))

    run_var = tk.StringVar(value="Run: -")
    ttk.Label(bottom_frame, textvariable=run_var).pack(side="left", padx=(0, UI_PADDING["wide"]))

    ttk.Button(bottom_frame, text="Run Simulation", command=lambda: start_simulation(sim_entries)).pack(side='left', padx=UI_PADDING["label"])
    ttk.Button(bottom_frame, text="Stop Simulation", command=stop_simulation).pack(side='left', padx=UI_PADDING["label"])


    # --- Tab 2: Additional settings ---
    early_stop_tab = create_scrollable_tab(notebook, "Advanced Settings")
    early_stop_tab.columnconfigure(0, weight=1)

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
        "- This overrides the single start temperature if enabled.\n"
        "- Number of Temperatures specifies how many temperatures to sample between Min and Max Temperature.\n"
        "- NOTE: Target temperature, ΔT, and ΔT steps settings still apply within each temperature."
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
    plots_tab = create_scrollable_tab(notebook, "Analysis")

    analysis_frame = ttk.Frame(plots_tab, style="Analysis.TFrame")
    analysis_frame.pack(fill="both", expand=True)
    analysis_frame.columnconfigure(0, weight=1)

    # Frame inside the tab to hold plot buttons
    plot_buttons_container = ttk.LabelFrame(analysis_frame)
    plot_buttons_label = ttk.Label(
        plot_buttons_container,
        text="Plots",
        style="FrameLabel.TLabel",
        padding=(UI_PADDING["frame_label"], 0, UI_PADDING["frame_label"], 0),
    )
    plot_buttons_container.configure(labelwidget=plot_buttons_label)
    plot_buttons_container.grid(row=0, column=0, sticky="nsew", padx=UI_PADDING["section"], pady=UI_PADDING["section"])
    plot_buttons_container.columnconfigure(0, weight=0)

    plot_buttons_frame = ttk.Frame(plot_buttons_container)
    plot_buttons_frame.pack(anchor="w", padx=UI_PADDING["label"], pady=UI_PADDING["label"])
    plot_buttons_frame.columnconfigure(0, weight=0)

    # Buttons for Energy / Magnetization plots (static)
    plot_energy_btn = ttk.Button(
        plot_buttons_frame,
        text="Plot Energy vs Step",
        command=lambda: plots.plot_energy_time(file_path=get_active_snapshot_file()),
        width=ANALYSIS_BTN_WIDTH,
        state="disabled"
    )
    plot_mag_btn = ttk.Button(
        plot_buttons_frame,
        text="Plot Magnetization vs Step",
        command=lambda: plots.plot_mag_time(file_path=get_active_snapshot_file()),
        width=ANALYSIS_BTN_WIDTH,
        state="disabled"
    )

    plot_thermo_btn = ttk.Button(
        plot_buttons_frame,
        text="Plot Thermodynamics",
        command=lambda: (lambda data: plots.plot_thermodynamics(data[0], data[1]))(plots.load_snapshots(get_active_snapshot_file())),
        width=ANALYSIS_BTN_WIDTH,
        state="disabled"
    )

    plot_energy_btn.grid(row=0, column=0, padx=UI_PADDING["label"], pady=UI_PADDING["label"], sticky="w")
    plot_mag_btn.grid(row=1, column=0, padx=UI_PADDING["label"], pady=UI_PADDING["label"], sticky="w")
    plot_thermo_btn.grid(row=2, column=0, padx=UI_PADDING["label"], pady=UI_PADDING["label"], sticky="w")

    thermo_status_var = tk.StringVar(value="Thermodynamics requires multiple temperatures.")
    ttk.Label(plot_buttons_frame, textvariable=thermo_status_var, wraplength=400).grid(
        row=3,
        column=0,
        padx=UI_PADDING["label"],
        pady=(0, UI_PADDING["label"]),
        sticky="w",
    )

    # --- Dynamic Visualization Buttons ---
    viz_container = ttk.LabelFrame(analysis_frame)
    viz_label = ttk.Label(
        viz_container,
        text="Visualize Snapshots",
        style="FrameLabel.TLabel",
        padding=(UI_PADDING["frame_label"], 0, UI_PADDING["frame_label"], 0),
    )
    viz_container.configure(labelwidget=viz_label)
    viz_container.grid(row=1, column=0, padx=UI_PADDING["section"], pady=UI_PADDING["section"], sticky="nsew")
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
    keyboard_label.grid(row=0, column=0, padx=UI_PADDING["label"], pady=(UI_PADDING["label"], UI_PADDING["label"]), sticky="w")

    viz_buttons_frame = ttk.Frame(viz_container)
    viz_buttons_frame.grid(row=1, column=0, sticky="nsew", padx=UI_PADDING["label"], pady=(0, UI_PADDING["label"]))

    # Initial population of snapshot buttons
    build_viz_buttons(get_active_snapshot_file(), viz_buttons_frame)

    # --- Saved Runs Selection ---
    runs_container = ttk.LabelFrame(analysis_frame)
    runs_label = ttk.Label(
        runs_container,
        text="Saved Runs",
        style="FrameLabel.TLabel",
        padding=(UI_PADDING["frame_label"], 0, UI_PADDING["frame_label"], 0),
    )
    runs_container.configure(labelwidget=runs_label)
    runs_container.grid(row=2, column=0, padx=UI_PADDING["section"], pady=UI_PADDING["section"], sticky="nsew")
    runs_container.columnconfigure(0, weight=1)

    toolbar = ttk.Frame(runs_container)
    toolbar.grid(row=0, column=0, sticky="ew", padx=UI_PADDING["label"], pady=UI_PADDING["label"])
    ttk.Button(toolbar, text="Refresh", command=populate_saved_runs, width=ANALYSIS_BTN_WIDTH//2).pack(side="left", padx=UI_PADDING["tight"])
    ttk.Button(toolbar, text="Select All", command=select_all_runs, width=ANALYSIS_BTN_WIDTH//2).pack(side="left", padx=UI_PADDING["tight"])
    ttk.Button(toolbar, text="Clear", command=clear_run_selection, width=ANALYSIS_BTN_WIDTH//2).pack(side="left", padx=UI_PADDING["tight"])
    ttk.Button(toolbar, text="Load Selected", command=combine_selected_runs, width=ANALYSIS_BTN_WIDTH//2).pack(side="left", padx=UI_PADDING["tight"])
    ttk.Button(toolbar, text="Delete Selected", command=delete_selected_runs, width=ANALYSIS_BTN_WIDTH//2).pack(side="left", padx=UI_PADDING["tight"])

    runs_status_var = tk.StringVar(value="")
    ttk.Label(runs_container, textvariable=runs_status_var, wraplength=600).grid(
        row=1,
        column=0,
        sticky="w",
        padx=UI_PADDING["label"],
        pady=(0, UI_PADDING["label"]),
    )

    ttk.Label(
        runs_container,
        text="Tip: Click column headers to sort. Use Ctrl/Shift to select multiple rows.",
        wraplength=600,
        justify="left"
    ).grid(row=2, column=0, sticky="w", padx=UI_PADDING["label"], pady=(0, UI_PADDING["label"]))

    runs_columns = ("T", "H", "W", "L", "J", "B", "dT", "dT_steps", "MC steps", "seed")
    runs_tree = ttk.Treeview(runs_container, columns=runs_columns, show="headings", selectmode="extended", height=8)
    for col in runs_columns:
        heading = "dT steps" if col == "dT_steps" else col
        runs_tree.heading(col, text=heading, command=lambda c=col: _sort_treeview(runs_tree, c, False))
        runs_tree.column(col, width=70, anchor="center")

    runs_tree.grid(row=3, column=0, sticky="nsew")
    runs_scrollbar = ttk.Scrollbar(runs_container, orient="vertical", command=runs_tree.yview)
    runs_scrollbar.grid(row=3, column=1, sticky="ns")
    runs_tree.configure(yscrollcommand=runs_scrollbar.set)

    runs_tree.bind("<Button-1>", lambda event: root.after(0, _tree_update_anchor))
    runs_tree.bind("<KeyRelease-Up>", lambda event: _tree_update_anchor() if not (event.state & 0x0001) else None)
    runs_tree.bind("<KeyRelease-Down>", lambda event: _tree_update_anchor() if not (event.state & 0x0001) else None)
    runs_tree.bind("<Shift-Up>", lambda event: _tree_shift_select(-1))
    runs_tree.bind("<Shift-Down>", lambda event: _tree_shift_select(1))
    populate_saved_runs()

    # --- Tab 4: Ising Model Info ---
    info_frame = create_scrollable_tab(notebook, "Ising Model Info")
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

    ttk.Label(info_frame, text=info_text, justify='left', wraplength=600).pack(
        padx=UI_PADDING["wide"],
        pady=UI_PADDING["wide"],
        anchor='w',
    )

    root.mainloop()

# ------------------------
# Run GUI
# ------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    build_gui()
