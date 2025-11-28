import math
import numpy as np
import matplotlib.pyplot as plt
import csv

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def cubic_bezier(t, P0, P1, P2, P3):
    """Cubic Bezier curve for a given t in [0,1]."""
    return ((1 - t)**3) * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + (t**3) * P3


def alpha_from_CL(CL_target, slope_per_deg, alpha0_deg):
    """
    Linear lift curve:
      CL = a * (alpha - alpha0)  ->  alpha = CL/a + alpha0
    slope_per_deg: dCL/d(alpha_deg)
    alpha, alpha0 in degrees.
    """
    return CL_target / slope_per_deg + alpha0_deg


def smooth_list(vals, w=0.3):
    """
    Light smoothing of a 1D array (moving average blend).
    Keeps endpoints fixed.
    """
    vals = np.asarray(vals, float)
    out = vals.copy()
    n = len(vals)
    for i in range(1, n - 1):
        out[i] = (1.0 - w) * vals[i] + 0.5 * w * (vals[i - 1] + vals[i + 1])
    return out


def cd_from_cl(cl, cl_opt=0.7, cd_min=0.015, k2=0.04):
    """
    Simple parabolic drag polar:
      CD = CD_min + k2 (CL - CL_opt)^2
    (Low-Re NACA 4412/0012-ish behaviour)
    """
    cl = np.asarray(cl, float)
    return cd_min + k2 * (cl - cl_opt)**2


# -----------------------------------------------------------
# Rotor design: free vortex + Bezier chord, AoA at max L/D
# -----------------------------------------------------------

def design_rotor_base(
    D=0.125,
    T_des=25.0,
    phi_mean=0.75,
    DF_limit=0.4,
    hub_to_tip=0.30,
    rho=1.225,
    psi_des=0.4,
    B_rotor=11,
    n_span=16,
    cR_min=0.04,
    cR_max=0.30,
    CL_target=0.9,
):
    """
    Base rotor design (before chord scaling) using:
      - Free-vortex swirl (Vθ2 ∝ 1/r),
      - Constant axial velocity (phi_mean = Vx/U at mean radius),
      - Bezier chord shape,
      - CL_target ~ 0.9 (near high L/D for NACA 4412),
      - DF_limit sets MINIMUM chord (to avoid excessive diffusion),
      - No separate 'slender' objective: chord will later be scaled by BEMT.
    """

    R_tip = 0.5 * D
    R_hub = hub_to_tip * R_tip
    A_annulus = math.pi * (R_tip**2 - R_hub**2)
    r_m = math.sqrt(0.5 * (R_hub**2 + R_tip**2))

    # Static momentum sizing: T = 0.5 rho A V4^2
    V4 = math.sqrt(2.0 * T_des / (rho * A_annulus))
    dh0 = 0.5 * V4**2

    # Stage loading psi = Δh0 / U^2 at mean radius
    U_m = math.sqrt(dh0 / psi_des)
    omega = U_m / r_m
    rpm = omega * 60.0 / (2.0 * math.pi)

    Vx_const = phi_mean * U_m
    print(Vx_const)

    # Radial grid
    r = np.linspace(R_hub * 1.03, R_tip * 0.99, n_span)
    r_R = r / R_tip

    U = omega * r
    Vx = np.full_like(r, Vx_const)
    Vtheta1 = np.zeros_like(r)
    Vtheta2 = dh0 / U             # free-vortex swirl
    V1 = np.sqrt(Vx**2 + Vtheta1**2)
    V2 = np.sqrt(Vx**2 + Vtheta2**2)

    beta1 = np.degrees(np.arctan2(Vx, U - Vtheta1))
    beta2 = np.degrees(np.arctan2(Vx, U - Vtheta2))

    # Aerodynamics: CL target ~0.9, AoA from linear lift curve
    a_rotor = 0.11
    alpha0_rotor = -2.0
    alpha_section = alpha_from_CL(CL_target, a_rotor, alpha0_rotor)
    CL = np.full_like(r, CL_target)
    alpha = np.full_like(r, alpha_section)

    # Bezier-based chord (smooth, not “slender-optimised”)
    cR_root = 0.18
    cR_tip  = 0.06
    cR1     = 0.16
    cR2     = 0.10

    t = (r - R_hub) / (R_tip - R_hub)
    c_pref_R = np.array([cubic_bezier(ti, cR_root, cR1, cR2, cR_tip) for ti in t])
    c_pref_R = np.clip(c_pref_R, cR_min, cR_max)
    c_pref = c_pref_R * R_tip

    # DF-based minimum chord (Lieblein)
    c_from_DF = np.zeros_like(r)
    DF_if_pref = np.zeros_like(r)
    for i, ri in enumerate(r):
        s = 2.0 * math.pi * ri / B_rotor
        if V1[i] <= 1e-6:
            continue
        s_over_c_pref = s / max(c_pref[i], 1e-9)
        DF_if_pref[i] = (
            1.0
            - V2[i] / V1[i]
            + (Vtheta2[i] - Vtheta1[i]) / (2.0 * V1[i]) * s_over_c_pref
        )
        dVtheta = Vtheta2[i] - Vtheta1[i]
        if abs(dVtheta) < 1e-8:
            continue
        s_over_c_req = (DF_limit - 1.0 + V2[i] / V1[i]) * (2.0 * V1[i] / dVtheta)
        if s_over_c_req > 0.0:
            c_from_DF[i] = s / s_over_c_req

    c = np.maximum(c_pref, c_from_DF)
    c = np.minimum(c, cR_max * R_tip)
    c = smooth_list(c, w=0.3)

    DF = np.zeros_like(r)
    for i, ri in enumerate(r):
        s = 2.0 * math.pi * ri / B_rotor
        if c[i] <= 0.0 or V1[i] <= 1e-6:
            continue
        s_over_c = s / c[i]
        DF[i] = (
            1.0
            - V2[i] / V1[i]
            + (Vtheta2[i] - Vtheta1[i]) / (2.0 * V1[i]) * s_over_c
        )

    # "Equivalent static" disc speed & momentum thrust
    V_disc_eff = 0.5 * V4
    T_mom = 0.5 * rho * A_annulus * V4**2
    P_static_eff = T_mom * V_disc_eff

    # Calculate drag coefficient (CD) from CL using a simple polar
    CD = cd_from_cl(CL, cl_opt=0.7, cd_min=0.015, k2=0.04)

    return {
        "r": r,
        "r_R": r_R,
        "c": c,
        "beta1": beta1,
        "beta2": beta2,
        "DF": DF,
        "rpm": rpm,
        "omega": omega,
        "U": U,
        "Vx": Vx,
        "Vtheta1": Vtheta1,
        "Vtheta2": Vtheta2,
        "R_tip": R_tip,
        "R_hub": R_hub,
        "A_annulus": A_annulus,
        "V4": V4,
        "V_disc_eff": V_disc_eff,
        "T_mom": T_mom,
        "CL": CL,
        "alpha": alpha,
        "B_rotor": B_rotor,
        "rho": rho,
        "P_static_eff": P_static_eff,
        "CD": CD,
        "CL_target": CL_target,
    }


# -----------------------------------------------------------
# Save output distributions to CSV
# -----------------------------------------------------------

def save_to_csv(rotor, file_name="rotor_distribution.csv"):
    """
    Save rotor distributions to a CSV file:
      - r/R (normalized radius)
      - c (chord)
      - beta (inflow angle)
      - alpha (AoA)
      - Pitch Angle (beta + alpha)
      - CL, CD
      - Leading edge (LE) sweep and LE height
    """
    header = ['r/R', 'Chord [m]', 'Inflow angle beta [deg]', 'AoA alpha [deg]', 'Pitch Angle [deg]', 'CL', 'CD', 'LE Sweep [deg]', 'LE Height [m]']
    rows = []

    # Calculate LE Sweep and LE Height
    for i in range(len(rotor['r_R'])):
        r_norm = rotor['r_R'][i]
        c = rotor['c'][i]
        beta = rotor['beta1'][i]
        alpha = rotor['alpha'][i]
        CL = rotor['CL'][i]
        CD = rotor['CD'][i]
        
        # Pitch Angle = Inflow Angle (beta) + Angle of Attack (alpha)
        pitch_angle = beta + alpha

        # For LE Sweep, assuming linear change in sweep along the blade
        # Note: r_norm * R_tip gives actual radius r.
        r_actual = r_norm * rotor['R_tip']
        
        # Calculate sweep. using numpy/math functions safely
        if r_actual > 0:
            LE_sweep = math.degrees(np.arctan(c / r_actual))
        else:
            LE_sweep = 0.0
            
        LE_height = r_actual * math.sin(math.radians(LE_sweep))

        rows.append([r_norm, c, beta, alpha, pitch_angle, CL, CD, LE_sweep, LE_height])

    # Write to CSV
    try:
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Rotor distributions saved to {file_name}")
    except IOError as e:
        print(f"Error saving CSV: {e}")


# -----------------------------------------------------------
# Main: run the design and save results
# -----------------------------------------------------------

if __name__ == "__main__":
    # Design parameters for the rotor
    D = 0.125      # rotor outer diameter [m]
    T_des = 25.0   # required thrust at design point [N]
    phi_mean = 0.75
    DF_limit = 0.4
    hub_to_tip = 0.30
    psi_des = 0.4
    B_rotor = 11
    n_span = 16
    CL_target = 0.9  # High CL for L/D optimization

    # Sea level conditions
    rho = 1.225  # kg/m^3 (sea level)

    # Design the rotor base
    rotor = design_rotor_base(
        D=D,
        T_des=T_des,
        phi_mean=phi_mean,
        DF_limit=DF_limit,
        hub_to_tip=hub_to_tip,
        rho=rho,
        psi_des=psi_des,
        B_rotor=B_rotor,
        n_span=n_span,
        cR_min=0.04,
        cR_max=0.30,
        CL_target=CL_target,
    )

    # Save the distributions to a CSV file
    save_to_csv(rotor)

    # Plot Chord and Pitch distributions
    plt.figure()
    plt.plot(rotor['r_R'], rotor['c'], label="Chord Distribution")
    plt.xlabel("r / R_tip")
    plt.ylabel("Chord [m]")
    plt.title("Chord Distribution (Rotor)")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(rotor['r_R'], rotor['beta1'], label="Metal Angle beta_m")
    plt.xlabel("r / R_tip")
    plt.ylabel("Metal angle [deg]")
    plt.title("Metal Angle Distribution (Rotor)")
    plt.grid(True)
    plt.legend()

    plt.show()