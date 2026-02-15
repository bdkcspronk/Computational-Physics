"""
Tkinter GUI for the lid-driven cavity solver.

Lets the user:
  - Enter simulation parameters (Re, nx, ny, bottom velocity, …).
  - Queue multiple parameter sets at once (ranges or single values).
  - Run all queued simulations (reusing cached results via hash).
  - Browse completed results and launch post-processing plots / animations.
"""

import sys
import pickle
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Make sure sibling modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from liddrivencavity import params_hash, result_path, RESULTS_DIR
import postprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_range(text):
    """Parse a parameter field that can be a single value or a range.

    Accepted formats:
        "100"          -> [100.0]
        "10, 20, 100"  -> [10.0, 20.0, 100.0]
        "10:10:100"    -> [10, 20, 30, …, 100]   (start:step:stop inclusive)
    """
    text = text.strip()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) == 3:
            import numpy as np
            start, step, stop = parts
            return list(np.arange(start, stop + step * 0.5, step))
        raise ValueError(f"Range must be start:step:stop, got '{text}'")
    return [float(v.strip()) for v in text.split(",")]


def _parse_int_range(text):
    return [int(v) for v in _parse_range(text)]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class CavityGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lid-Driven Cavity Solver")
        self.geometry("900x700")
        self.resizable(True, True)

        self._build_input_frame()
        self._build_queue_frame()
        self._build_results_frame()
        self._build_animation_frame()
        self._build_log_frame()

        self._queue: list[dict] = []
        self._stop_queue_flag = False
        self._current_process: subprocess.Popen | None = None
        self._current_hash: str | None = None
        self._refresh_results()

    # ---- UI construction --------------------------------------------------

    def _build_input_frame(self):
        frame = ttk.LabelFrame(self, text="Simulation Parameters", padding=8)
        frame.pack(fill="x", padx=8, pady=(8, 4))

        labels = [
            ("Re", "100"),
            ("Grid size (n)", "41"),
            ("Lid velocity", "1.0"),
            ("Bottom velocity", "0.0"),
            ("rel_tol", "1e-6"),
            ("Max steps", "100000"),
            ("Save every N", "100"),
            ("CFL factor", "1.0"),
        ]
        self._entries: dict[str, ttk.Entry] = {}
        for col, (label, default) in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=0, column=col, padx=4)
            entry = ttk.Entry(frame, width=14)
            entry.insert(0, default)
            entry.grid(row=1, column=col, padx=4, pady=2)
            self._entries[label] = entry

        ttk.Label(
            frame,
            text="Tip: use comma-separated values (10,20,100) or ranges (10:10:100) for Re, Grid size, and Bottom velocity. Grid is always square (nx=ny=n).",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=len(labels), sticky="w", pady=(4, 0))

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=len(labels), pady=6)
        ttk.Button(btn_frame, text="Add to Queue", command=self._add_to_queue).pack(
            side="left", padx=4
        )
        ttk.Button(btn_frame, text="Run Queue", command=self._run_queue).pack(
            side="left", padx=4
        )
        ttk.Button(btn_frame, text="Stop Queue", command=self._stop_queue).pack(
            side="left", padx=4
        )
        ttk.Button(btn_frame, text="Clear Queue", command=self._clear_queue).pack(
            side="left", padx=4
        )

    def _build_queue_frame(self):
        frame = ttk.LabelFrame(self, text="Queue", padding=4)
        frame.pack(fill="both", padx=8, pady=4, expand=False)

        cols = ("Re", "n", "Lid vel", "Bot vel", "tol", "Max steps",
                "Save N", "Hash", "Status")
        self._queue_tree = ttk.Treeview(frame, columns=cols, show="headings",
                                        height=5)
        for c in cols:
            self._queue_tree.heading(c, text=c)
            self._queue_tree.column(c, width=80, anchor="center")
        self._queue_tree.pack(fill="both", expand=True)

    def _build_results_frame(self):
        frame = ttk.LabelFrame(self, text="Saved Results", padding=4)
        frame.pack(fill="both", padx=8, pady=4, expand=True)

        # Buttons at the top
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side="top", fill="x", pady=(0, 4))
        ttk.Button(btn_frame, text="View Plots",
                   command=self._view_plots).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="View Animation",
                   command=self._view_animation).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Delete",
                   command=self._delete_result).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Refresh",
                   command=self._refresh_results).pack(side="left", padx=4)

        # Tree and scrollbar below
        cols = ("Hash", "Re", "n", "Lid vel", "Bot vel", "tol",
                "Max steps", "Snapshots", "Converged")
        self._res_tree = ttk.Treeview(frame, columns=cols, show="headings",
                                      height=6)
        for c in cols:
            self._res_tree.heading(c, text=c, command=lambda col=c: self._sort_results(col))
            self._res_tree.column(c, width=80, anchor="center")

        scrollbar = ttk.Scrollbar(frame, orient="vertical",
                                  command=self._res_tree.yview)
        self._res_tree.configure(yscrollcommand=scrollbar.set)
        self._res_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Store data for sorting
        self._res_data = []
        self._res_sort_col = None
        self._res_sort_reverse = False

    def _build_animation_frame(self):
        """Animation field selector tab."""
        frame = ttk.LabelFrame(self, text="Animate Fields", padding=8)
        frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(frame, text="Field:").grid(row=0, column=0, sticky="w", padx=4)
        self._field_var = tk.StringVar(value="speed")
        field_combo = ttk.Combobox(
            frame,
            textvariable=self._field_var,
            values=["speed", "vorticity", "streamfunction", "pressure", "u", "v"],
            state="readonly",
            width=20,
        )
        field_combo.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(frame, text="FPS:").grid(row=0, column=2, sticky="w", padx=4)
        self._fps_entry = ttk.Entry(frame, width=8)
        self._fps_entry.insert(0, "10")
        self._fps_entry.grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(frame, text="Max Frames:").grid(row=0, column=4, sticky="w", padx=4)
        self._max_frames_entry = ttk.Entry(frame, width=8)
        self._max_frames_entry.insert(0, "0")
        self._max_frames_entry.grid(row=0, column=5, sticky="w", padx=4)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=6, pady=6)
        ttk.Button(
            btn_frame,
            text="Generate Animation",
            command=self._generate_field_animation,
        ).pack(side="left", padx=4)

    def _build_log_frame(self):
        frame = ttk.LabelFrame(self, text="Log", padding=4)
        frame.pack(fill="both", padx=8, pady=(4, 8), expand=False)

        self._log_text = tk.Text(frame, height=6, state="disabled",
                                 wrap="word")
        self._log_text.pack(fill="both", expand=True)

    # ---- Logging ----------------------------------------------------------

    def _log(self, msg):
        try:
            self._log_text.configure(state="normal")
            self._log_text.insert("end", msg + "\n")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        except Exception as e:
            print(f"Log error: {e}")  # Fallback to console

    # ---- Queue management -------------------------------------------------

    def _add_to_queue(self):
        try:
            re_vals = _parse_range(self._entries["Re"].get())
            n_vals = _parse_int_range(self._entries["Grid size (n)"].get())
            lid_vals = _parse_range(self._entries["Lid velocity"].get())
            bot_vals = _parse_range(self._entries["Bottom velocity"].get())
            tol_vals = _parse_range(self._entries["rel_tol"].get())
            max_vals = _parse_int_range(self._entries["Max steps"].get())
            save_vals = _parse_int_range(self._entries["Save every N"].get())
            cfl_factor = float(self._entries["CFL factor"].get())
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        count = 0
        for re in re_vals:
            for n in n_vals:
                for lid in lid_vals:
                    for bot in bot_vals:
                        for tol in tol_vals:
                            for ms in max_vals:
                                for sn in save_vals:
                                    h = params_hash(re, n, n, lid,
                                                    bot, tol, ms, sn, cfl_factor)
                                    item = {
                                        "re": re, "nx": n, "ny": n,
                                        "lid_velocity": lid,
                                        "bottom_velocity": bot,
                                        "rel_tol": tol,
                                        "max_steps": ms,
                                        "save_every": sn,
                                        "cfl_factor": cfl_factor,
                                        "hash": h,
                                    }
                                    self._queue.append(item)
                                    cached = result_path(h).exists()
                                    status = "cached" if cached else "pending"
                                    self._queue_tree.insert(
                                        "", "end",
                                        values=(
                                            re, n, lid, bot,
                                            tol, ms, sn, h, status,
                                        ),
                                    )
                                    count += 1
        self._log(f"Added {count} case(s) to queue.")

    def _clear_queue(self):
        self._queue.clear()
        for item in self._queue_tree.get_children():
            self._queue_tree.delete(item)
        self._log("Queue cleared.")

    def _run_queue(self):
        if not self._queue:
            self._log("Queue is empty.")
            return
        self._stop_queue_flag = False
        # Run in a background thread so the UI stays responsive
        thread = threading.Thread(target=self._run_queue_worker, daemon=True)
        thread.start()

    def _stop_queue(self):
        if self._current_process is not None:
            # Kill the running subprocess
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._current_process.kill()
                self._current_process.wait()
            self._log(f"Killed running process (hash: {self._current_hash})")
            
            # Remove partial output file
            if self._current_hash:
                partial_path = result_path(self._current_hash)
                if partial_path.exists():
                    try:
                        partial_path.unlink()
                        self._log(f"Removed partial file: {partial_path.name}")
                    except Exception as e:
                        self._log(f"Failed to remove {partial_path.name}: {e}")
            
            self._current_process = None
            self._current_hash = None
        
        self._stop_queue_flag = True
        self._log("Queue stopped.")

    def _run_queue_worker(self):
        total = len(self._queue)
        children = self._queue_tree.get_children()
        for idx, item in enumerate(self._queue):
            # Check stop flag at start of each iteration
            if self._stop_queue_flag:
                self.after(0, self._log, "Queue stopped by user.")
                break

            iid = children[idx]

            # skip if already cached
            if result_path(item["hash"]).exists():
                self._queue_tree.set(iid, "Status", "cached")
                self.after(0, self._log,
                           f"[{idx+1}/{total}] {item['hash']} — cached")
                self.after(50, lambda iid=iid: self._queue_tree.delete(iid))
                continue

            self._queue_tree.set(iid, "Status", "running…")
            self.after(0, self._log,
                       f"[{idx+1}/{total}] Running {item['hash']}…")

            try:
                # Store current hash for potential cleanup
                self._current_hash = item["hash"]
                
                # Run solver as subprocess so we can kill it
                cmd = [
                    sys.executable, "-c",
                    f"""from liddrivencavity import run_simulation
run_simulation(re={item['re']}, nx={item['nx']}, ny={item['ny']}, 
               lid_velocity={item['lid_velocity']}, bottom_velocity={item['bottom_velocity']}, 
               rel_tol={item['rel_tol']}, max_steps={item['max_steps']}, 
               save_every={item['save_every']}, cfl_factor={item['cfl_factor']})"""
                ]
                self._current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                                         stderr=subprocess.PIPE)
                stdout, stderr = self._current_process.communicate()
                
                if self._current_process.returncode != 0:
                    raise RuntimeError(f"Solver failed: {stderr.decode()}")
                
                self._current_process = None
                self._current_hash = None
                self._queue_tree.set(iid, "Status", "done")
                self.after(0, self._log,
                           f"[{idx+1}/{total}] {item['hash']} — done")
                self.after(50, lambda iid=iid: self._queue_tree.delete(iid))
            except Exception as exc:
                self._current_process = None
                self._current_hash = None
                self._queue_tree.set(iid, "Status", "error")
                self.after(0, self._log,
                           f"[{idx+1}/{total}] {item['hash']} — ERROR: {exc}")

        self.after(0, self._refresh_results)
        self.after(0, self._log, "Queue finished.")
        self._queue = []  # Clear queue list after run completes
        self._current_process = None
        self._current_hash = None

    # ---- Results browser --------------------------------------------------

    def _refresh_results(self):
        for item in self._res_tree.get_children():
            self._res_tree.delete(item)

        self._res_data = []

        if not RESULTS_DIR.exists():
            return

        for pkl in sorted(RESULTS_DIR.glob("*.pkl")):
            try:
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
                p = data["params"]
                row = (
                    data.get("hash", pkl.stem),
                    p["re"], p["nx"],
                    p["lid_velocity"], p["bottom_velocity"],
                    p["rel_tol"], p["max_steps"],
                    len(data["snapshots"]),
                    data.get("converged_step", "—"),
                )
                self._res_data.append((str(pkl), row))
            except Exception:
                pass

        self._populate_results_tree()

    def _populate_results_tree(self):
        """Populate the tree view from _res_data."""
        for item in self._res_tree.get_children():
            self._res_tree.delete(item)
        for path, row in self._res_data:
            self._res_tree.insert("", "end", iid=path, values=row)

    def _sort_results(self, col):
        """Sort results by the given column."""
        col_idx = {
            "Hash": 0, "Re": 1, "n": 2,
            "Lid vel": 3, "Bot vel": 4, "tol": 5,
            "Max steps": 6, "Snapshots": 7, "Converged": 8
        }.get(col, 0)

        # Toggle sort order if clicking the same column
        if self._res_sort_col == col:
            self._res_sort_reverse = not self._res_sort_reverse
        else:
            self._res_sort_col = col
            self._res_sort_reverse = False

        # Try to sort numerically if possible, otherwise sort as string
        try:
            self._res_data.sort(
                key=lambda x: float(x[1][col_idx]) if x[1][col_idx] != "—" else float('-inf'),
                reverse=self._res_sort_reverse
            )
        except (ValueError, TypeError):
            self._res_data.sort(
                key=lambda x: str(x[1][col_idx]),
                reverse=self._res_sort_reverse
            )

        self._populate_results_tree()

    def _selected_result_path(self):
        sel = self._res_tree.selection()
        if not sel:
            messagebox.showinfo("No selection",
                                "Select a result row first.")
            return None
        return Path(sel[0])

    def _view_plots(self):
        path = self._selected_result_path()
        if path is None:
            return
        self._log(f"Opening plots for {path.stem}…")
        data = postprocess.load_results(path)
        postprocess.plot_final_state(data, show=True)

    def _view_animation(self):
        path = self._selected_result_path()
        if path is None:
            return
        self._log(f"Generating animation for {path.stem}…")
        data = postprocess.load_results(path)
        gif_path = str(path.with_suffix(".gif"))
        postprocess.animate_speed(data, output_path=gif_path, show=True)
        self._log(f"Animation saved to {gif_path}")

    def _delete_result(self):
        path = self._selected_result_path()
        if path is None:
            return
        if messagebox.askyesno("Delete", f"Delete {path.name}?"):
            path.unlink()
            self._log(f"Deleted {path.name}")
            self._refresh_results()

    def _generate_field_animation(self):
        path = self._selected_result_path()
        if path is None:
            return
        try:
            fps = int(self._fps_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "FPS must be an integer")
            return
        try:
            max_frames_str = self._max_frames_entry.get()
            max_frames = None if max_frames_str == "0" or max_frames_str == "" else int(max_frames_str)
        except ValueError:
            messagebox.showerror("Input Error", "Max Frames must be an integer (or 0 for all)")
            return
        field = self._field_var.get()
        self._log(f"Generating {field} animation for {path.stem}…")
        data = postprocess.load_results(path)
        gif_path = str(path.with_stem(f"{path.stem}_{field}"))
        postprocess.animate_field(data, field_name=field, output_path=gif_path,
                                  fps=fps, show=True, max_frames=max_frames)
        self._log(f"Animation saved to {gif_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = CavityGUI()
    app.mainloop()
