"""
Lid-driven cavity solver.

Runs a single simulation and saves snapshots of the stream function and
vorticity to a pickle file every *save_every* steps.  Results are stored
under ``results/<hash>.pkl`` where the hash is derived from the physical
and numerical parameters so that identical runs are automatically reused.

Can be used as an importable module or run directly from the command line::

    python liddrivencavity.py --re 100 --nx 41 --ny 41 --bottom_velocity 0
"""

import numpy as np
import pickle
import hashlib
from pathlib import Path
from numba import njit


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def vorticity_transport(u, v, vort, dx, dy, re, dt):
    ny, nx = vort.shape
    vort_new = vort.copy()
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            dw_dx = (vort[i, j + 1] - vort[i, j - 1]) / (2.0 * dx)
            dw_dy = (vort[i + 1, j] - vort[i - 1, j]) / (2.0 * dy)
            d2w_dx2 = (vort[i, j + 1] - 2.0 * vort[i, j] + vort[i, j - 1]) / (dx * dx)
            d2w_dy2 = (vort[i + 1, j] - 2.0 * vort[i, j] + vort[i - 1, j]) / (dy * dy)
            vort_new[i, j] = vort[i, j] + dt * (
                -(u[i, j] * dw_dx + v[i, j] * dw_dy)
                + (1.0 / re) * (d2w_dx2 + d2w_dy2)
            )
    return vort_new


@njit(cache=True)
def streamfunction_sor(psi, vort, dx, dy, iterations, omega):
    ny, nx = psi.shape
    dx2 = dx * dx
    dy2 = dy * dy
    coef = 1.0 / (2.0 * (dx2 + dy2))
    for _ in range(iterations):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                rhs = (
                    (psi[i, j + 1] + psi[i, j - 1]) * dy2
                    + (psi[i + 1, j] + psi[i - 1, j]) * dx2
                    + vort[i, j] * dx2 * dy2
                ) * coef
                psi[i, j] = (1.0 - omega) * psi[i, j] + omega * rhs
    return psi


@njit(cache=True)
def recover_velocity(psi, u, v, dx, dy):
    ny, nx = psi.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            u[i, j] = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dy)
            v[i, j] = -(psi[i, j + 1] - psi[i, j - 1]) / (2.0 * dx)


# ---------------------------------------------------------------------------
# Utility: parameter hashing
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"


def params_hash(re, nx, ny, lid_velocity, bottom_velocity, rel_tol, max_steps, save_every, cfl_factor):
    """Return an 8-character hex hash that uniquely identifies the run."""
    key = f"{re}_{nx}_{ny}_{lid_velocity}_{bottom_velocity}_{rel_tol}_{max_steps}_{save_every}_{cfl_factor}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def result_path(run_hash):
    """Return the Path for a given run hash."""
    return RESULTS_DIR / f"{run_hash}.pkl"


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class LidDrivenCavity:
    def __init__(self, nx=41, ny=41, re=100, lid_velocity=1.0, bottom_velocity=0.0):
        self.nx = nx
        self.ny = ny
        self.re = re
        self.lid_velocity = lid_velocity
        self.bottom_velocity = bottom_velocity

        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)

        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.psi = np.zeros((ny, nx))
        self.vort = np.zeros((ny, nx))

        self._apply_bc()

    # -- boundary conditions ------------------------------------------------

    def _apply_bc(self):
        self.u[-1, :] = self.lid_velocity
        self.u[0, :] = self.bottom_velocity
        self.u[:, 0] = 0
        self.u[:, -1] = 0
        self.v[0, :] = 0
        self.v[-1, :] = 0
        self.v[:, 0] = 0
        self.v[:, -1] = 0

    def _vorticity_bc(self):
        self.psi[0, :] = 0
        self.psi[-1, :] = 0
        self.psi[:, 0] = 0
        self.psi[:, -1] = 0
        self.vort[-1, 1:-1] = (
            -2.0 * (self.psi[-2, 1:-1] - self.psi[-1, 1:-1]) / (self.dy ** 2)
            - 2.0 * self.lid_velocity / self.dy
        )
        self.vort[0, 1:-1] = (
            -2.0 * (self.psi[1, 1:-1] - self.psi[0, 1:-1]) / (self.dy ** 2)
            + 2.0 * self.bottom_velocity / self.dy
        )
        self.vort[1:-1, 0] = (
            -2.0 * (self.psi[1:-1, 1] - self.psi[1:-1, 0]) / (self.dx ** 2)
        )
        self.vort[1:-1, -1] = (
            -2.0 * (self.psi[1:-1, -2] - self.psi[1:-1, -1]) / (self.dx ** 2)
        )

    # -- sub-solvers --------------------------------------------------------

    def _solve_vorticity_transport(self, dt):
        self.vort = vorticity_transport(
            self.u, self.v, self.vort, self.dx, self.dy, self.re, dt,
        )
        self._vorticity_bc()

    def _solve_streamfunction(self, iterations=200, sor_omega=1.7):
        self.psi = streamfunction_sor(
            self.psi, self.vort, self.dx, self.dy, iterations, sor_omega,
        )
        self._vorticity_bc()

    def _recover_velocity(self):
        recover_velocity(self.psi, self.u, self.v, self.dx, self.dy)

    def _stable_dt(self, cfl=0.5):
        umax = np.max(np.abs(self.u))
        vmax = np.max(np.abs(self.v))
        dt_adv = np.inf
        if umax > 0:
            dt_adv = min(dt_adv, cfl * self.dx / umax)
        if vmax > 0:
            dt_adv = min(dt_adv, cfl * self.dy / vmax)
        nu = 1.0 / self.re
        dt_diff = 1.0 / (2.0 * nu * (1.0 / self.dx ** 2 + 1.0 / self.dy ** 2))
        return min(dt_adv, dt_diff)

    # -- main solve ---------------------------------------------------------

    def solve(self, max_steps=100000, rel_tol=1e-6, save_every=100,
              callback=None, cfl_factor=1.0):
        """
        Run the solver.

        Parameters
        ----------
        max_steps : int
            Maximum number of time steps.
        rel_tol : float
            Relative convergence tolerance on vorticity.
        save_every : int
            Save a snapshot every this many steps.
        callback : callable or None
            ``callback(step, converged)`` called every *save_every* steps and
            at convergence.  Return ``False`` from callback to abort early.
        cfl_factor : float
            Multiplier for CFL condition (1.0=default, >1.0 allows larger steps).

        Returns
        -------
        snapshots : list[dict]
            Each entry has keys ``step``, ``psi``, ``vort``.
        converged_step : int or None
            Step at which convergence was reached, or None.
        """
        eps = np.finfo(float).eps
        snapshots = []
        converged_step = None

        self._apply_bc()

        for step in range(max_steps):
            vort_prev = self.vort.copy()
            dt = self._stable_dt(cfl=0.5 * cfl_factor)
            self._solve_vorticity_transport(dt)
            self._solve_streamfunction()
            self._recover_velocity()
            self._apply_bc()

            # convergence check
            diff = np.linalg.norm(self.vort - vort_prev)
            denom = max(np.linalg.norm(self.vort), eps)
            rel_diff = diff / denom
            converged = rel_diff < rel_tol

            # snapshot
            if step % save_every == 0 or converged:
                snapshots.append({
                    "step": step,
                    "psi": self.psi.copy(),
                    "vort": self.vort.copy(),
                })
                if callback is not None:
                    if callback(step, converged) is False:
                        break

            if converged:
                converged_step = step
                print(
                    f"Converged at step {step} (rel_diff={rel_diff:.3e})"
                )
                break

        return snapshots, converged_step


# ---------------------------------------------------------------------------
# Run + save
# ---------------------------------------------------------------------------

def run_simulation(
    re=100,
    nx=41,
    ny=41,
    lid_velocity=1.0,
    bottom_velocity=0.0,
    rel_tol=1e-6,
    max_steps=100000,
    save_every=100,
    force=False,
    callback=None,
    cfl_factor=1.0,
):
    """
    Run a single simulation and persist results to disk.

    Returns (run_hash, output_path, already_existed).
    """
    run_hash = params_hash(re, nx, ny, lid_velocity, bottom_velocity,
                           rel_tol, max_steps, save_every, cfl_factor)
    out = result_path(run_hash)

    if out.exists() and not force:
        print(f"Results already exist: {out}  (hash {run_hash})")
        return run_hash, out, True

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cavity = LidDrivenCavity(nx=nx, ny=ny, re=re,
                             lid_velocity=lid_velocity,
                             bottom_velocity=bottom_velocity)
    snapshots, converged_step = cavity.solve(
        max_steps=max_steps,
        rel_tol=rel_tol,
        save_every=save_every,
        callback=callback,
        cfl_factor=cfl_factor,
    )

    data = {
        "params": {
            "re": re,
            "nx": nx,
            "ny": ny,
            "lid_velocity": lid_velocity,
            "bottom_velocity": bottom_velocity,
            "rel_tol": rel_tol,
            "max_steps": max_steps,
            "save_every": save_every,
        },
        "x": cavity.x,
        "y": cavity.y,
        "snapshots": snapshots,
        "converged_step": converged_step,
        "hash": run_hash,
    }

    with open(out, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved {len(snapshots)} snapshots to {out}  (hash {run_hash})")
    return run_hash, out, False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lid-driven cavity solver")
    parser.add_argument("--re", type=float, default=100)
    parser.add_argument("--nx", type=int, default=41)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--lid_velocity", type=float, default=1.0)
    parser.add_argument("--bottom_velocity", type=float, default=0.0)
    parser.add_argument("--rel_tol", type=float, default=1e-6)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    args = parser.parse_args()

    run_simulation(
        re=args.re,
        nx=args.nx,
        ny=args.ny,
        lid_velocity=args.lid_velocity,
        bottom_velocity=args.bottom_velocity,
        rel_tol=args.rel_tol,
        max_steps=args.max_steps,
        save_every=args.save_every,
        force=args.force,
    )