from __future__ import annotations

import argparse
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "result.txt"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figs"

# Coma cluster center
RA0 = 194.935  # deg
DEC0 = 27.935  # deg

# Coma systemic redshift
ZC = 0.0231

# Speed of light
C_KM_S = 299792.458  # km/s

# Task 1 criteria
RMAX = 1.5  # deg
ZMIN = 0.017
ZMAX = 0.028

# Task 2 sigma-clipping criteria
SEED_WINDOW_KMS = 3000.0
CLIP_SIGMA = 3.0
MAX_ITER = 20


NUMERIC_COLUMNS = {
    "No.",
    "RA",
    "DEC",
    "Velocity",
    "Redshift",
    "Separation",
    "References",
    "Notes",
    "Photometry Points",
    "Positions",
    "Redshift Points",
    "Diameter Points",
    "Associations",
}


def nanmean(values: pd.Series | np.ndarray) -> float:
    return float(np.nanmean(np.asarray(values, dtype=float)))


def nanstd_sample(values: pd.Series | np.ndarray) -> float:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[~np.isnan(values_array)]
    if values_array.size <= 1:
        return float("nan")
    return float(np.std(values_array, ddof=1))


def read_ned_table(infile: Path) -> pd.DataFrame:
    raw_lines = infile.read_text(encoding="utf-8", errors="replace").splitlines()

    header_idx = next(
        (
            idx
            for idx, line in enumerate(raw_lines)
            if line.strip().startswith("No.|Object Name|RA|DEC|Type|Velocity|Redshift")
        ),
        None,
    )
    if header_idx is None:
        raise ValueError("Cannot find NED table header in result.txt.")

    header_line = raw_lines[header_idx].strip()
    data_lines = [
        line.strip()
        for line in raw_lines[header_idx + 1 :]
        if re.match(r"^\s*\d+\|", line)
    ]

    table_text = "\n".join([header_line, *data_lines])
    table = pd.read_csv(StringIO(table_text), sep="|", dtype=str)
    table.columns = [column.strip() for column in table.columns]

    for column in table.columns:
        table[column] = table[column].astype(str).str.strip()

    for column in NUMERIC_COLUMNS.intersection(table.columns):
        table[column] = pd.to_numeric(table[column], errors="coerce")

    return table


def compute_membership(table: pd.DataFrame) -> dict[str, pd.Series | np.ndarray]:
    ra = table["RA"]
    dec = table["DEC"]
    z = table["Redshift"]
    typ = table["Type"].astype(str)

    valid = ra.notna() & dec.notna() & z.notna()
    is_galaxy = valid & typ.str.strip().str.casefold().eq("g")

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    ra0_rad = np.deg2rad(RA0)
    dec0_rad = np.deg2rad(DEC0)

    cosang = (
        np.sin(dec0_rad) * np.sin(dec_rad)
        + np.cos(dec0_rad) * np.cos(dec_rad) * np.cos(ra_rad - ra0_rad)
    )
    cosang = np.clip(cosang, -1.0, 1.0)
    r_deg = np.rad2deg(np.arccos(cosang))

    v_rel = C_KM_S * (z - ZC) / (1.0 + ZC)

    is_spatial_sample = is_galaxy & (r_deg <= RMAX)
    is_task1_member = is_spatial_sample & (z >= ZMIN) & (z <= ZMAX)

    is_sigma_member = is_spatial_sample & (v_rel.abs() <= SEED_WINDOW_KMS)
    sigma_iterations: list[dict[str, float | int]] = []

    for iteration in range(1, MAX_ITER + 1):
        old_mask = is_sigma_member.copy()
        v_now = v_rel[is_sigma_member].dropna()

        mu_now = nanmean(v_now)
        sig_now = nanstd_sample(v_now)
        lower_now = mu_now - CLIP_SIGMA * sig_now
        upper_now = mu_now + CLIP_SIGMA * sig_now

        is_sigma_member = is_spatial_sample & (v_rel >= lower_now) & (v_rel <= upper_now)

        sigma_iterations.append(
            {
                "iteration": iteration,
                "n": int(is_sigma_member.sum()),
                "mean": mu_now,
                "sigma": sig_now,
                "lower": lower_now,
                "upper": upper_now,
            }
        )

        if old_mask.equals(is_sigma_member):
            break

    v_sigma = v_rel[is_sigma_member].dropna()
    mu_sigma = nanmean(v_sigma)
    sig_sigma = nanstd_sample(v_sigma)
    final_lower = mu_sigma - CLIP_SIGMA * sig_sigma
    final_upper = mu_sigma + CLIP_SIGMA * sig_sigma

    is_overlap = is_task1_member & is_sigma_member
    is_task1_only = is_task1_member & ~is_sigma_member
    is_sigma_only = is_sigma_member & ~is_task1_member

    return {
        "ra": ra,
        "dec": dec,
        "z": z,
        "is_galaxy": is_galaxy,
        "r_deg": r_deg,
        "v_rel": v_rel,
        "is_spatial_sample": is_spatial_sample,
        "is_task1_member": is_task1_member,
        "is_sigma_member": is_sigma_member,
        "sigma_iterations": sigma_iterations,
        "mu_sigma": mu_sigma,
        "sig_sigma": sig_sigma,
        "final_lower": final_lower,
        "final_upper": final_upper,
        "is_overlap": is_overlap,
        "is_task1_only": is_task1_only,
        "is_sigma_only": is_sigma_only,
    }


def print_report(results: dict[str, pd.Series | np.ndarray]) -> None:
    z = results["z"]
    v_rel = results["v_rel"]
    is_spatial_sample = results["is_spatial_sample"]
    is_task1_member = results["is_task1_member"]
    is_sigma_member = results["is_sigma_member"]
    sigma_iterations = results["sigma_iterations"]

    print("\n===== Task 1: redshift-window selection =====")
    print(f"Spatial sample galaxies, R <= {RMAX:.2f} deg: {int(is_spatial_sample.sum())}")
    print(f"Task 1 redshift-window members: {int(is_task1_member.sum())}")
    print(f"Mean z of Task 1 members: {nanmean(z[is_task1_member]):.6f}")
    print(f"Std z of Task 1 members : {nanstd_sample(z[is_task1_member]):.6f}")
    print(f"Mean v_rel of Task 1 members: {nanmean(v_rel[is_task1_member]):.2f} km/s")
    print(f"Std v_rel of Task 1 members : {nanstd_sample(v_rel[is_task1_member]):.2f} km/s")

    print("\n===== Task 2: 3-sigma clipping =====")
    print(f"Initial seed: |v_rel| <= {SEED_WINDOW_KMS:.0f} km/s")

    for item in sigma_iterations:
        print(
            f"Iter {item['iteration']:2d}: "
            f"N = {item['n']:4d}, "
            f"mean = {item['mean']:8.2f} km/s, "
            f"sigma = {item['sigma']:8.2f} km/s, "
            f"limits = [{item['lower']:8.2f}, {item['upper']:8.2f}]"
        )

    if sigma_iterations:
        print(f"Sigma-clipping converged at iteration {sigma_iterations[-1]['iteration']}.")

    print(f"\nFinal Sigma-clipping members: {int(is_sigma_member.sum())}")
    print(f"Final mean v_rel: {results['mu_sigma']:.2f} km/s")
    print(f"Final sigma v_rel: {results['sig_sigma']:.2f} km/s")
    print(
        f"Final 3-sigma limits: "
        f"[{results['final_lower']:.2f}, {results['final_upper']:.2f}] km/s"
    )

    print("\n===== Numbers for report table =====")
    print(f"Task 1 redshift-window members : {int(is_task1_member.sum())}")
    print(f"3-sigma clipping members       : {int(is_sigma_member.sum())}")
    print(f"Overlap members                : {int(results['is_overlap'].sum())}")
    print(f"Task 1 only                    : {int(results['is_task1_only'].sum())}")
    print(f"3-sigma clipping only          : {int(results['is_sigma_only'].sum())}")


def save_figures(results: dict[str, pd.Series | np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ra = results["ra"]
    dec = results["dec"]
    z = results["z"]
    is_galaxy = results["is_galaxy"]
    r_deg = results["r_deg"]
    v_rel = results["v_rel"]
    is_spatial_sample = results["is_spatial_sample"]
    is_task1_member = results["is_task1_member"]
    is_sigma_member = results["is_sigma_member"]

    non_task1_galaxies = is_galaxy & ~is_task1_member

    fig, ax = plt.subplots(figsize=(8.5, 7.0), facecolor="white")
    ax.scatter(ra[non_task1_galaxies], dec[non_task1_galaxies], s=10, c=[[0.72, 0.72, 0.72]])
    ax.scatter(ra[is_task1_member], dec[is_task1_member], s=16, c="red")
    ax.plot(RA0, DEC0, "kp", markersize=15, markerfacecolor="yellow", linewidth=1.2)

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    circle_dec = DEC0 + RMAX * np.sin(theta)
    circle_ra = RA0 + RMAX * np.cos(theta) / np.cos(np.deg2rad(DEC0))
    ax.plot(circle_ra, circle_dec, "k--", linewidth=1.3)

    ax.invert_xaxis()
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Spatial distribution of Coma candidate and selected member galaxies")
    ax.legend(
        [
            "Non-member galaxies in broad sample",
            "Selected member galaxies",
            "Coma center",
            "R = 1.5 deg selection radius",
        ],
        loc="lower left",
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_ra_dec_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 6.5), facecolor="white")
    ax.scatter(r_deg[non_task1_galaxies], z[non_task1_galaxies], s=10, c=[[0.72, 0.72, 0.72]])
    ax.scatter(r_deg[is_task1_member], z[is_task1_member], s=16, c="red")
    ax.axvline(RMAX, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(ZMIN, color="red", linestyle="--", linewidth=1.3)
    ax.axhline(ZC, color="black", linestyle="-", linewidth=1.5)
    ax.axhline(ZMAX, color="red", linestyle="--", linewidth=1.3)

    ax.set_xlabel("Angular distance from Coma center (deg)")
    ax.set_ylabel("Redshift z")
    ax.set_title("Membership selection in radius-redshift space")
    ax.legend(
        [
            "Non-member galaxies in broad sample",
            "Selected member galaxies",
            "Radius limit",
            "Redshift limits",
            "Cluster systemic redshift",
        ],
        loc="upper right",
    )
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(0.005, 0.060)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_radius_redshift_selection.png", dpi=200)
    plt.close(fig)

    v_spatial = v_rel[is_spatial_sample].dropna()
    v_sig = v_rel[is_sigma_member].dropna()
    bins = np.arange(
        np.floor(min(v_spatial.min(), v_sig.min()) / 200.0) * 200.0,
        np.ceil(max(v_spatial.max(), v_sig.max()) / 200.0) * 200.0 + 200.0,
        200.0,
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.0), facecolor="white")
    ax.hist(v_spatial, bins=bins, color=[0.82, 0.82, 0.82], edgecolor="none")
    ax.hist(v_sig, bins=bins, color=[1.0, 0.35, 0.35], edgecolor="none")
    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.6)
    ax.axvline(results["final_lower"], color="red", linestyle="--", linewidth=1.5)
    ax.axvline(results["final_upper"], color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Line-of-sight velocity relative to Coma center (km/s)")
    ax.set_ylabel("Number of galaxies")
    ax.set_title("Task 2: 3-sigma clipping in velocity space")
    ax.legend(
        [
            "Spatial sample",
            "Final Sigma-clipping members",
            "Coma systemic velocity",
            "Final 3-sigma limits",
        ],
        loc="upper right",
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_sigma_clipping_histogram.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reimplementation of mission1_2.m for Coma cluster member selection."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to NED result.txt. Defaults to data/result.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help="Directory where PNG figures are saved. Defaults to figs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = read_ned_table(args.input)
    results = compute_membership(table)
    print_report(results)
    save_figures(results, args.output_dir)
    print(f"\nSaved figures to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
