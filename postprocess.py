"""
Post-processing for lid-driven cavity simulation results.

Loads a pickle file produced by ``liddrivencavity.py``, recovers the
velocity field and (optionally) the pressure field from the stored stream
function and vorticity snapshots, and generates plots or animations.

Usage::

    python postprocess.py results/abc12345.pkl          # static plots of final state
    python postprocess.py results/abc12345.pkl --animate # speed-magnitude GIF
    python postprocess.py results/abc12345.pkl --all     # everything
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit


# ---------------------------------------------------------------------------
# Velocity recovery (same kernel used by the solver)
# ---------------------------------------------------------------------------

@njit(cache=True)
def recover_velocity(psi, dx, dy):
    ny, nx = psi.shape
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            u[i, j] = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dy)
            v[i, j] = -(psi[i, j + 1] - psi[i, j - 1]) / (2.0 * dx)
    return u, v


@njit(cache=True)
def _pressure_poisson_rhs(u, v, dx, dy, re):
    """Build the RHS of the pressure Poisson equation."""
    ny, nx = u.shape
    rhs = np.zeros((ny, nx))
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            du_dx = (u[i, j + 1] - u[i, j - 1]) / (2.0 * dx)
            du_dy = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dy)
            dv_dx = (v[i, j + 1] - v[i, j - 1]) / (2.0 * dx)
            dv_dy = (v[i + 1, j] - v[i - 1, j]) / (2.0 * dy)
            rhs[i, j] = -(du_dx ** 2 + 2.0 * du_dy * dv_dx + dv_dy ** 2)
    return rhs


@njit(cache=True)
def _solve_pressure_sor(rhs, dx, dy, iterations=2000, omega=1.7):
    """Solve ∇²p = rhs with p=0 on all boundaries via SOR."""
    ny, nx = rhs.shape
    p = np.zeros((ny, nx))
    dx2 = dx * dx
    dy2 = dy * dy
    coef = 1.0 / (2.0 * (dx2 + dy2))
    for _ in range(iterations):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                val = (
                    (p[i, j + 1] + p[i, j - 1]) * dy2
                    + (p[i + 1, j] + p[i - 1, j]) * dx2
                    - rhs[i, j] * dx2 * dy2
                ) * coef
                p[i, j] = (1.0 - omega) * p[i, j] + omega * val
    return p


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def load_results(path):
    """Load a pickle results file and return the data dict."""
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_velocity(psi, dx, dy, lid_velocity=1.0, bottom_velocity=0.0):
    """Recover velocity from stream function and apply wall BCs."""
    u, v = recover_velocity(psi, dx, dy)
    u[-1, :] = lid_velocity
    u[0, :] = bottom_velocity
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    return u, v


def compute_pressure(u, v, dx, dy, re):
    """Solve the pressure Poisson equation from a velocity field."""
    rhs = _pressure_poisson_rhs(u, v, dx, dy, re)
    return _solve_pressure_sor(rhs, dx, dy)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_final_state(data, show=True):
    """Plot the converged (last-snapshot) fields."""
    params = data["params"]
    x = data["x"]
    y = data["y"]
    snap = data["snapshots"][-1]
    psi = snap["psi"]
    vort = snap["vort"]

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    u, v = compute_velocity(psi, dx, dy,
                            lid_velocity=params["lid_velocity"],
                            bottom_velocity=params["bottom_velocity"])
    vel_mag = np.sqrt(u ** 2 + v ** 2)
    pressure = compute_pressure(u, v, dx, dy, params["re"])

    title_suffix = (
        f"Re={params['re']}, nx={params['nx']}, "
        f"Ubot={params['bottom_velocity']}"
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window') and hasattr(mng.window, 'showMaximized'):
            mng.window.showMaximized()
    except Exception:
        pass

    # velocity magnitude
    im0 = axes[0, 0].contourf(x, y, vel_mag, levels=20)
    axes[0, 0].set_title("Velocity Magnitude")
    fig.colorbar(im0, ax=axes[0, 0])

    # vorticity
    im1 = axes[0, 1].contourf(x, y, vort, levels=20)
    axes[0, 1].set_title("Vorticity")
    fig.colorbar(im1, ax=axes[0, 1])

    # stream function
    im2 = axes[0, 2].contourf(x, y, psi, levels=20)
    axes[0, 2].set_title("Stream Function")
    fig.colorbar(im2, ax=axes[0, 2])

    # pressure
    im3 = axes[1, 0].contourf(x, y, pressure, levels=20)
    axes[1, 0].set_title("Pressure")
    fig.colorbar(im3, ax=axes[1, 0])

    # midline velocity profiles
    ny, nx_grid = psi.shape
    mid_y = ny // 2
    mid_x = nx_grid // 2
    axes[1, 1].plot(x, u[mid_y, :], label="u(x, 0.5)")
    axes[1, 1].plot(y, v[:, mid_x], label="v(0.5, y)")
    axes[1, 1].set_title("Midline Velocity Profiles")
    axes[1, 1].legend()

    # velocity vectors
    skip = max(1, nx_grid // 20)
    axes[1, 2].quiver(
        x[::skip], y[::skip], u[::skip, ::skip], v[::skip, ::skip],
    )
    axes[1, 2].set_title("Velocity Vectors")
    axes[1, 2].set_aspect("equal")

    fig.suptitle(title_suffix)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    return fig


def animate_field(data, field_name="speed", output_path=None, fps=10, show=True, max_frames=None):
    """
    Create an animation of a field over time.
    
    Parameters
    ----------
    data : dict
        The results dictionary from load_results().
    field_name : str
        One of: "speed", "vorticity", "streamfunction", "pressure", "u", "v".
    output_path : str or None
        Path to save the GIF. If None, derives from data["_source_path"].
    fps : int
        Frames per second.
    show : bool
        Whether to display the animation.
    max_frames : int or None
        Maximum number of frames to animate. If None, animate all frames.
    """
    params = data["params"]
    x = data["x"]
    y = data["y"]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    snapshots = data["snapshots"]
    
    # Limit snapshots if max_frames is specified
    if max_frames is not None and len(snapshots) > max_frames:
        snapshots = snapshots[:max_frames]

    # Pre-compute the field for all snapshots
    fields = []
    for snap in snapshots:
        psi = snap["psi"]
        vort = snap["vort"]
        
        if field_name == "vorticity":
            fields.append(vort)
        elif field_name == "streamfunction":
            fields.append(psi)
        else:
            # Need to recover velocity for speed, u, v, pressure
            u, v = compute_velocity(psi, dx, dy,
                                    lid_velocity=params["lid_velocity"],
                                    bottom_velocity=params["bottom_velocity"])
            if field_name == "speed":
                fields.append(np.sqrt(u ** 2 + v ** 2))
            elif field_name == "u":
                fields.append(u)
            elif field_name == "v":
                fields.append(v)
            elif field_name == "pressure":
                fields.append(compute_pressure(u, v, dx, dy, params["re"]))
            else:
                raise ValueError(f"Unknown field: {field_name}")

    vmin = min(np.min(f) for f in fields)
    vmax = max(np.max(f) for f in fields)

    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window') and hasattr(mng.window, 'showMaximized'):
            mng.window.showMaximized()
    except Exception:
        pass
    
    im = ax.imshow(
        fields[0],
        origin="lower",
        extent=(0, 1, 0, 1),
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=field_name.capitalize())
    title = ax.set_title(
        f"{field_name.capitalize()} | Frame 1/{len(fields)} | Step {snapshots[0]['step']}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame):
        im.set_data(fields[frame])
        title.set_text(
            f"{field_name.capitalize()} | Frame {frame+1}/{len(fields)} | Step {snapshots[frame]['step']}"
        )
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(fields), interval=1000 // fps, blit=True,
    )

    if output_path is None:
        suffix = f"_{field_name}.gif"
        output_path = str(
            Path(data.get("_source_path", "animation")).with_suffix(suffix)
        )
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    print(f"Animation saved to {output_path}")

    if show:
        plt.show()
    return anim


def animate_speed(data, output_path=None, fps=10, show=True):
    """Create a speed-magnitude animation from all snapshots."""
    return animate_field(data, field_name="speed", output_path=output_path,
                         fps=fps, show=show)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-process lid-driven cavity results",
    )
    parser.add_argument("result_file", type=str,
                        help="Path to a .pkl results file")
    parser.add_argument("--animate", action="store_true",
                        help="Generate a speed-magnitude animation")
    parser.add_argument("--all", action="store_true",
                        help="Generate all plots and animation")
    parser.add_argument("--gif", type=str, default=None,
                        help="Output path for the animation GIF")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open interactive plot windows")
    args = parser.parse_args()

    data = load_results(args.result_file)
    data["_source_path"] = args.result_file
    show = not args.no_show

    if args.all or not args.animate:
        plot_final_state(data, show=show)

    if args.all or args.animate:
        gif_path = args.gif
        if gif_path is None:
            gif_path = str(Path(args.result_file).with_suffix(".gif"))
        animate_speed(data, output_path=gif_path, show=show)


if __name__ == "__main__":
    main()
