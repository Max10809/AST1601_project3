from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mission1_2 import C_KM_S, ZC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "data" / "task3_common_members.csv"
FIG_DIR = PROJECT_ROOT / "figs"

H0 = 70.0
G = 4.30091e-9

M_SUN_R = 4.65
UPSILON_R = 3.0


def parse_magnitude(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    match = re.match(r"^([0-9]*\.?[0-9]+)", text)
    if match is None:
        return float("nan")
    return float(match.group(1))


def angular_sep_rad(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)

    cosang = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)


def pairwise_projected_distances(ra, dec, da_mpc):
    n = len(ra)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            theta = angular_sep_rad(ra[i], dec[i], ra[j], dec[j])
            rij = da_mpc * theta
            if rij > 0:
                distances.append(rij)

    return np.array(distances)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    df["Redshift z"] = pd.to_numeric(df["Redshift z"], errors="coerce")
    df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
    df["DEC"] = pd.to_numeric(df["DEC"], errors="coerce")
    df["Magnitude"] = df["Magnitude"].map(parse_magnitude)

    df_task4 = df.dropna(subset=["RA", "DEC", "Redshift z"]).copy()
    df_task5 = df_task4.dropna(subset=["Magnitude"]).copy()

    n = len(df_task4)
    n_luminous = len(df_task5)
    z_mean = df_task4["Redshift z"].mean()

    v_los = C_KM_S * (df_task4["Redshift z"].to_numpy() - ZC) / (1.0 + ZC)
    sigma_v = np.std(v_los, ddof=1)

    d_l_mpc = C_KM_S * z_mean / H0
    d_a_mpc = d_l_mpc / (1.0 + z_mean) ** 2

    ra = df_task4["RA"].to_numpy()
    dec = df_task4["DEC"].to_numpy()

    rij = pairwise_projected_distances(ra, dec, d_a_mpc)
    r_pv = n * (n - 1) / np.sum(1.0 / rij)

    m_vir = (3.0 * np.pi / (2.0 * G)) * sigma_v**2 * r_pv

    distance_modulus = 5.0 * np.log10(d_l_mpc * 1e6 / 10.0)
    abs_mag = df_task5["Magnitude"].to_numpy() - distance_modulus

    luminosity = 10.0 ** (-0.4 * (abs_mag - M_SUN_R))
    l_total = np.sum(luminosity)
    m_luminous = UPSILON_R * l_total

    print("===== Task 4: Zwicky / Virial Mass Estimate =====")
    print(f"Number of member galaxies N = {n}")
    print(f"Cluster systemic redshift z_c = {ZC:.6f}")
    print(f"Mean redshift z_mean = {z_mean:.6f}")
    print(f"Luminosity distance D_L = {d_l_mpc:.2f} Mpc")
    print(f"Angular diameter distance D_A = {d_a_mpc:.2f} Mpc")
    print(f"Velocity dispersion sigma_v = {sigma_v:.2f} km/s")
    print(f"Projected virial radius R_PV = {r_pv:.3f} Mpc")
    print(f"Virial mass M_vir = {m_vir:.3e} Msun")

    print("\n===== Task 5: Luminous Matter Mass Estimate =====")
    print(f"Galaxies with usable magnitude N_mag = {n_luminous}")
    print(f"Galaxies excluded due to missing magnitude = {n - n_luminous}")
    print(f"Assumed solar absolute magnitude M_sun,r = {M_SUN_R:.2f}")
    print(f"Assumed mass-to-light ratio Upsilon_r = {UPSILON_R:.2f} Msun/Lsun")
    print(f"Total luminosity L_total = {l_total:.3e} Lsun")
    print(f"Luminous matter mass M_lum = {m_luminous:.3e} Msun")

    plt.figure(figsize=(8, 5))
    plt.hist(v_los, bins=20, edgecolor="black")
    plt.axvline(0.0, linestyle="--")
    plt.axvline(sigma_v, linestyle=":")
    plt.axvline(-sigma_v, linestyle=":")
    plt.xlabel("Line-of-sight velocity relative to Coma center (km/s)")
    plt.ylabel("Number of galaxies")
    plt.title("Task 4: Velocity distribution of member galaxies")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task4_velocity_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(ra, dec, s=18)
    plt.gca().invert_xaxis()
    plt.xlabel("RA (deg)")
    plt.ylabel("DEC (deg)")
    plt.title("Task 4: Projected distribution of member galaxies")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task4_projected_distribution.png", dpi=200)
    plt.close()

    luminosity_sorted = np.sort(luminosity)[::-1]
    cumulative_luminosity = np.cumsum(luminosity_sorted)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumulative_luminosity) + 1), cumulative_luminosity)
    plt.xlabel("Number of brightest galaxies included")
    plt.ylabel("Cumulative luminosity (Lsun)")
    plt.title("Task 5: Cumulative luminosity of member galaxies")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task5_cumulative_luminosity.png", dpi=200)
    plt.close()

    output = pd.DataFrame(
        {
            "Quantity": [
                "N_members",
                "N_with_magnitude",
                "z_c",
                "z_mean",
                "D_L_Mpc",
                "D_A_Mpc",
                "sigma_v_km_s",
                "R_PV_Mpc",
                "M_vir_Msun",
                "L_total_Lsun",
                "Upsilon_r",
                "M_luminous_Msun",
            ],
            "Value": [
                n,
                n_luminous,
                ZC,
                z_mean,
                d_l_mpc,
                d_a_mpc,
                sigma_v,
                r_pv,
                m_vir,
                l_total,
                UPSILON_R,
                m_luminous,
            ],
        }
    )

    output.to_csv(PROJECT_ROOT / "data" / "task4_5_mass_results.csv", index=False)
    print(f"\nSaved results to {PROJECT_ROOT / 'data' / 'task4_5_mass_results.csv'}")
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()