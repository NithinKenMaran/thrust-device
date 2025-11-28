import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Density from altitude using your Lagrange interpolation
# (altitude in meters, polynomial in km)
# -----------------------------------------------------------

def rho_from_alt_m(alt_m):
    """
    Density interpolation from altitude using the polynomial
    fitted through:
      (0 km, 1.225), (2 km, 1.007), (4 km, 0.8193)
    ρ(h_km) = 0.0037875 h^2 - 0.116575 h + 1.225
    """
    h_km = alt_m / 1000.0
    return 0.0037875*h_km**2 - 0.116575*h_km + 1.225


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
# Rotor base design (no “slender” objective)
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
      - DF_limit sets MINIMUM chord (avoid excessive diffusion).
    """

    R_tip = 0.5 * D
    R_hub = hub_to_tip * R_tip
    A_annulus = math.pi * (R_tip**2 - R_hub**2)
    r_m = math.sqrt(0.5 * (R_hub**2 + R_tip**2))

    # Δh0 = T / (ρ A)  (independent of freestream speed)
    dh0 = T_des / (rho * A_annulus)

    # Equivalent "static" far-wake speed V4_eff such that 0.5 V4^2 = Δh0
    V4_eff = math.sqrt(2.0 * dh0)

    # Stage loading psi = Δh0 / U^2 at mean radius
    U_m = math.sqrt(dh0 / psi_des)
    omega = U_m / r_m
    rpm = omega * 60.0 / (2.0 * math.pi)

    # Set axial velocity via φ_mean at mean radius
    Vx_const = phi_mean * U_m

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
        s_pitch = 2.0 * math.pi * ri / B_rotor
        if V1[i] <= 1e-6:
            continue
        s_over_c_pref = s_pitch / max(c_pref[i], 1e-9)
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
            c_from_DF[i] = s_pitch / s_over_c_req

    c = np.maximum(c_pref, c_from_DF)
    c = np.minimum(c, cR_max * R_tip)
    c = smooth_list(c, w=0.3)

    DF = np.zeros_like(r)
    for i, ri in enumerate(r):
        s_pitch = 2.0 * math.pi * ri / B_rotor
        if c[i] <= 0.0 or V1[i] <= 1e-6:
            continue
        s_over_c = s_pitch / c[i]
        DF[i] = (
            1.0
            - V2[i] / V1[i]
            + (Vtheta2[i] - Vtheta1[i]) / (2.0 * V1[i]) * s_over_c
        )

    # "Equivalent static" disc speed & momentum thrust
    V_disc_eff = 0.5 * V4_eff
    T_mom = 0.5 * rho * A_annulus * V4_eff**2
    P_static_eff = T_mom * V_disc_eff

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
        "V4_eff": V4_eff,
        "V_disc_eff": V_disc_eff,
        "T_mom": T_mom,
        "CL": CL,
        "alpha": alpha,
        "B_rotor": B_rotor,
        "rho": rho,
        "P_static_eff": P_static_eff,
        "CL_target": CL_target,
    }


# -----------------------------------------------------------
# BEMT thrust & torque (static-like internal flow) 
# -----------------------------------------------------------

def bemt_thrust_and_torque(rotor_res, cl_opt=None):
    rho = rotor_res["rho"]
    r = rotor_res["r"]
    c = rotor_res["c"]
    beta1_deg = rotor_res["beta1"]
    CL = rotor_res["CL"]
    if cl_opt is None:
        cl_opt = np.mean(CL)
    CD = cd_from_cl(CL, cl_opt=cl_opt, cd_min=0.015, k2=0.04)
    B = rotor_res["B_rotor"]
    U = rotor_res["U"]
    Vx = rotor_res["Vx"]

    dr = np.diff(r)
    r_mid = 0.5 * (r[:-1] + r[1:])
    U_mid = 0.5 * (U[:-1] + U[1:])
    Vx_mid = 0.5 * (Vx[:-1] + Vx[1:])
    c_mid = 0.5 * (c[:-1] + c[1:])
    CL_mid = 0.5 * (CL[:-1] + CL[1:])
    CD_mid = 0.5 * (CD[:-1] + CD[1:])
    beta_mid = 0.5 * (beta1_deg[:-1] + beta1_deg[1:])

    phi = np.radians(beta_mid)
    W_mid = np.sqrt(U_mid**2 + Vx_mid**2)

    dL = 0.5 * rho * (W_mid**2) * c_mid * CL_mid
    dD = 0.5 * rho * (W_mid**2) * c_mid * CD_mid

    dT = B * (dL * np.cos(phi) - dD * np.sin(phi)) * dr
    dQ = B * r_mid * (dL * np.sin(phi) + dD * np.cos(phi)) * dr

    T_bemt = np.sum(dT)
    Q_bemt = np.sum(dQ)
    D_profile_rel = np.sum(B * dD * dr)
    return {
        "T_bemt": T_bemt,
        "Q_bemt": Q_bemt,
        "D_profile_rel": D_profile_rel,
    }


# -----------------------------------------------------------
# Improved rotor: chord scaled so BEMT thrust = T_des (internal design)
# -----------------------------------------------------------

def design_rotor_with_scaling(**kwargs):
    """
    1) Build base rotor from momentum + CL_target + Bezier + DF_limit.
    2) Compute BEMT thrust for that rotor.
    3) Uniformly scale chord so BEMT thrust matches T_des.
    4) Recompute DF for scaled chord.
    """
    base = design_rotor_base(**kwargs)
    bemt_base = bemt_thrust_and_torque(base, cl_opt=base["CL_target"])
    T_bemt0 = bemt_base["T_bemt"]
    T_des = kwargs.get("T_des", 25.0)

    if T_bemt0 <= 1e-6:
        scale_c = 1.0
    else:
        scale_c = T_des / T_bemt0  # T ∝ c for fixed CL, velocities

    c_scaled = base["c"] * scale_c

    # Recompute DF for scaled chord
    r = base["r"]
    B = base["B_rotor"]
    Vx = base["Vx"]
    U = base["U"]
    Vtheta1 = base["Vtheta1"]
    Vtheta2 = base["Vtheta2"]
    V1 = np.sqrt(Vx**2 + Vtheta1**2)
    V2 = np.sqrt(Vx**2 + Vtheta2**2)
    DF_scaled = np.zeros_like(r)
    for i, ri in enumerate(r):
        s_pitch = 2.0 * math.pi * ri / B
        if c_scaled[i] <= 0.0 or V1[i] <= 1e-6:
            continue
        s_over_c = s_pitch / c_scaled[i]
        DF_scaled[i] = (
            1.0
            - V2[i] / V1[i]
            + (Vtheta2[i] - Vtheta1[i]) / (2.0 * V1[i]) * s_over_c
        )

    rotor = base.copy()
    rotor["c"] = c_scaled
    rotor["DF_scaled"] = DF_scaled
    rotor["scale_c"] = scale_c

    bemt_scaled = bemt_thrust_and_torque(rotor, cl_opt=base["CL_target"])
    rotor["T_bemt"] = bemt_scaled["T_bemt"]
    rotor["Q_bemt"] = bemt_scaled["Q_bemt"]
    rotor["D_profile_rel"] = bemt_scaled["D_profile_rel"]
    return rotor


# -----------------------------------------------------------
# Stator design (swirl removal, chord set by DF & drag)
# -----------------------------------------------------------

def design_stator_optimized(rotor_res, DF_limit=0.4, B_stator=13):
    rho = rotor_res["rho"]
    r = rotor_res["r"]
    R_tip = rotor_res["R_tip"]
    Vx_in = rotor_res["Vx"]
    Vtheta_in = rotor_res["Vtheta2"]

    V1s = np.sqrt(Vx_in**2 + Vtheta_in**2)
    beta1s = np.degrees(np.arctan2(Vx_in, Vtheta_in))

    Vtheta_out = np.zeros_like(r)
    Vx_out = Vx_in.copy()
    V2s = np.sqrt(Vx_out**2 + Vtheta_out**2)

    # CL ~ 0.4 for a stator (low-lift, low-loss)
    CL_target = 0.4
    a_stator = 0.10
    alpha0_stator = 0.0
    alpha_section = alpha_from_CL(CL_target, a_stator, alpha0_stator)
    CL = np.full_like(r, CL_target)
    alpha = np.full_like(r, alpha_section)

    c_from_DF = np.zeros_like(r)
    DF = np.zeros_like(r)
    for i, ri in enumerate(r):
        s_pitch = 2.0 * math.pi * ri / B_stator
        V1i = V1s[i]
        V2i = V2s[i]
        Vt1 = Vtheta_in[i]
        Vt2 = Vtheta_out[i]
        if V1i <= 1e-6:
            continue
        dVtheta = Vt2 - Vt1
        if abs(dVtheta) < 1e-8:
            continue
        s_over_c_req = (DF_limit - 1.0 + V2i / V1i) * (2.0 * V1i / dVtheta)
        if s_over_c_req > 0.0:
            c_from_DF[i] = s_pitch / s_over_c_req

    c_rotor = rotor_res["c"]
    c_pref = 0.8 * c_rotor  # stator a bit more slender to cut drag
    c = np.maximum(c_pref, c_from_DF)
    c = smooth_list(c, w=0.3)

    for i, ri in enumerate(r):
        s_pitch = 2.0 * math.pi * ri / B_stator
        V1i = V1s[i]
        V2i = V2s[i]
        Vt1 = Vtheta_in[i]
        Vt2 = Vtheta_out[i]
        ci = c[i]
        if ci <= 0.0 or V1i <= 1e-6:
            DF[i] = 0.0
            continue
        s_over_c = s_pitch / ci
        DF[i] = 1.0 - V2i / V1i + (Vt2 - Vt1) / (2.0 * V1i) * s_over_c

    CD = cd_from_cl(CL, cl_opt=0.4, cd_min=0.012, k2=0.03)
    W = V1s

    dr = np.diff(r)
    W_mid = 0.5 * (W[:-1] + W[1:])
    c_mid = 0.5 * (c[:-1] + c[1:])
    CD_mid = 0.5 * (CD[:-1] + CD[1:])

    Dp = 0.5 * rho * (W_mid**2) * c_mid * CD_mid
    D_profile = np.sum(Dp * dr) * B_stator

    # Estimate Residual Swirl using Carter's Rule (Deviation)
    # Turning angle theta = 90 - beta1s (Target is 90 deg/Axial from tangential)
    theta_turning = 90.0 - beta1s
    # Solidity sigma = B * c / (2 pi r)
    solidity = (B_stator * c) / (2.0 * np.pi * r)
    # Deviation delta = m * theta * sqrt(1/sigma) (m factor approx 0.23)
    delta_dev = 0.23 * theta_turning * np.sqrt(1.0 / np.maximum(solidity, 1e-9))
    # Residual Swirl Vtheta = Vx * tan(delta) (Deviation from axial direction)
    Vtheta_residual = Vx_out * np.tan(np.radians(delta_dev))

    return {
        "r": r,
        "r_R": r / R_tip,
        "c": c,
        "beta1": beta1s,
        "beta2": np.full_like(beta1s, 90.0),
        "DF": DF,
        "CL": CL,
        "CD": CD,
        "alpha": alpha,
        "D_profile": D_profile,
        "B_stator": B_stator,
        "Vtheta_residual": Vtheta_residual,
    }


# -----------------------------------------------------------
# Ideal actuator-disc at constant shaft power
# -----------------------------------------------------------

def actuator_disc_constant_power(P_shaft, A, V_inf, rho, tol=1e-6, max_iter=80):
    """
    Ideal actuator-disc with constant shaft power P_shaft:
      T  = 2 ρ A v_i (V∞ + v_i)
      P  = T (V∞ + v_i) = 2 ρ A v_i (V∞ + v_i)^2
    Solve for v_i, then get T and mass flow.
    """
    if rho <= 0 or A <= 0 or P_shaft <= 0:
        return 0.0, 0.0, 0.0, V_inf, V_inf

    def f(v):
        return 2.0 * rho * A * v * (V_inf + v)**2 - P_shaft

    low = 0.0
    base = (P_shaft / (2.0 * rho * A))**(1.0 / 3.0)
    high = max(1.0, 5.0 * base, 2.0 * V_inf + 5.0)

    f_high = f(high)
    it = 0
    while f_high <= 0 and it < 20:
        high *= 2.0
        f_high = f(high)
        it += 1
    if f_high <= 0:
        return 0.0, 0.0, 0.0, V_inf, V_inf

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            break
        if f_mid > 0:
            high = mid
        else:
            low = mid
    v_i = 0.5 * (low + high)

    V2 = V_inf + v_i
    V4 = V_inf + 2.0 * v_i
    T = 2.0 * rho * A * v_i * (V_inf + v_i)
    m_dot = rho * A * V2
    return v_i, T, m_dot, V2, V4


# -----------------------------------------------------------
# Design power from required T and V_inf (takeoff design point)
# -----------------------------------------------------------

def design_power_from_T_V(T_des, rho, A, V_inf):
    """
    From actuator disc:
      Thrust: T = 2 ρ A v (V∞ + v)
    Solve for v, then:
      V2 = V∞ + v
      V4 = V∞ + 2 v
      P  = T * V2
    """
    if rho <= 0 or A <= 0:
        return 0.0, 0.0, V_inf, V_inf, 0.0
    disc = V_inf**2 + 2.0*T_des/(rho*A)
    v_i = (-V_inf + math.sqrt(disc)) / 2.0
    V2 = V_inf + v_i
    V4 = V_inf + 2.0*v_i
    P = T_des * V2
    m_dot = rho * A * V2
    return P, v_i, V2, V4, m_dot


# -----------------------------------------------------------
# Main: takeoff = design point, static/cruise = off-design
# -----------------------------------------------------------

if __name__ == "__main__":
    # ---------- Parameters ----------
    D = 0.125      # rotor outer diameter [m]
    T_des = 25.0   # required thrust at design point [N]
    phi_mean = 0.75
    DF_limit = 0.4
    hub_to_tip = 0.30
    psi_des = 0.4
    B_rotor = 11
    B_stator = 13

    # Operating points
    V_static = 0.0         # off-design
    V_design = 20.0        # TAKEOFF (design point)
    V_cruise = 80.0        # off-design
    alt_static_m = 0.0
    alt_design_m = 0.0
    alt_cruise_m = 10000.0 * 0.3048  # 10,000 ft in meters

    rho0 = rho_from_alt_m(0.0)
    rho_cruise = rho_from_alt_m(alt_cruise_m)

    # ---------- Rotor (BEMT-scaled) ----------
    rotor = design_rotor_with_scaling(
        D=D,
        T_des=T_des,
        phi_mean=phi_mean,
        DF_limit=DF_limit,
        hub_to_tip=hub_to_tip,
        rho=rho0,
        psi_des=psi_des,
        B_rotor=B_rotor,
        n_span=20,
        cR_min=0.04,
        cR_max=0.30,
        CL_target=0.9,
    )

    # ---------- Stator ----------
    stator = design_stator_optimized(rotor, DF_limit=DF_limit, B_stator=B_stator)

    # Disc area (annulus)
    A = rotor["A_annulus"]
    R_tip = rotor["R_tip"]
    R_hub = rotor["R_hub"]

    # ---------- Design point (takeoff) power ----------
    P0, v_i_des, V2_des, V4_des, mdot_des = design_power_from_T_V(T_des, rho0, A, V_design)

    # ---------- Off-design (static & cruise) at constant P0 ----------
    v_i_static, T_static, mdot_static, V2_static, V4_static = actuator_disc_constant_power(
        P0, A, V_static, rho0
    )
    v_i_cruise, T_cruise, mdot_cruise, V2_cruise, V4_cruise = actuator_disc_constant_power(
        P0, A, V_cruise, rho_cruise
    )

    beta_m_rotor = rotor["beta1"] - rotor["alpha"]
    beta_m_stator = stator["beta1"] - stator["alpha"]
    r_R_rot = rotor["r_R"]
    r_R_sta = stator["r_R"]

    # ================= SWIRL CALCULATION FOR ALL CASES =================
    
    # 1. Design (Takeoff)
    # The 'stator' dict already calculated deviation for the Design condition 
    # since design_stator_optimized uses the 'rotor' dict (Takeoff conditions)
    # Vtheta_resid_des is available directly if we calculate it properly or assume 'stator' is design case
    Vtheta_resid_des = stator['Vtheta_residual']

    # 2. Static (V_inf = 0)
    # We need to estimate the rotor outflow swirl Vtheta2 at static
    # Approximation: Vtheta2 ~ (P0 / (rho * A * V2)) / U
    # Then re-calculate deviation using same stator geometry
    # Note: U is same (RPM constant assumption? Or P constant?)
    # P_static ~ P_design => RPM will vary slightly if we hold P constant, or Torque varies.
    # Let's assume RPM is roughly constant for simplicity or scale U by (P_avail/P_req)
    # For simplicity, let's use the 'rotor' U but scale Vtheta2 by relative power/flow.
    # A better approx: Vtheta2 = Delta_h0 / U. Delta_h0 = P / mdot.
    
    # Static:
    dh0_static = P0 / mdot_static
    # Assume RPM is similar or re-calculate. Let's use design Omega for first order.
    U = rotor['U'] 
    Vtheta2_static = dh0_static / U
    Vx_static_local = V2_static # Axial velocity
    beta1_static = np.degrees(np.arctan2(Vx_static_local, Vtheta2_static))
    theta_turning_static = 90.0 - beta1_static # Flow angle entering stator relative to tangential
    # Recalculate deviation
    solidity = (B_stator * stator['c']) / (2.0 * np.pi * stator['r'])
    delta_static = 0.23 * theta_turning_static * np.sqrt(1.0 / np.maximum(solidity, 1e-9))
    Vtheta_resid_static = Vx_static_local * np.tan(np.radians(delta_static))

    # 3. Cruise (V_inf = 80)
    dh0_cruise = P0 / mdot_cruise
    Vtheta2_cruise = dh0_cruise / U
    Vx_cruise_local = V2_cruise
    beta1_cruise = np.degrees(np.arctan2(Vx_cruise_local, Vtheta2_cruise))
    theta_turning_cruise = 90.0 - beta1_cruise
    delta_cruise = 0.23 * theta_turning_cruise * np.sqrt(1.0 / np.maximum(solidity, 1e-9))
    Vtheta_resid_cruise = Vx_cruise_local * np.tan(np.radians(delta_cruise))


    # ================= PRINT SUMMARY =================

    print("========== EDF Analytical Design ==========\n")
    print("Design point: TAKEOFF (V_inf = 20 m/s at sea level)")
    print("Off-design points: STATIC (0 m/s), CRUISE (80 m/s @ 10,000 ft)\n")

    print("Geometry / setup:")
    print(f"  D (outer diameter)    : {D*1000:.1f} mm")
    print(f"  T_des (takeoff)       : {T_des:.2f} N")
    print(f"  hub/tip ratio         : {hub_to_tip:.3f}")
    print(f"  B_rotor / B_stator    : {B_rotor} / {B_stator}")
    print(f"  ρ_sea-level (poly)    : {rho0:.4f} kg/m^3\n")

    print("Rotor (BEMT-optimised):")
    print(f"  Momentum Δh0 (T/(ρA)) : {T_des/(rho0*A):.1f} J/kg")
    print(f"  BEMT thrust T_bemt    : {rotor['T_bemt']:.3f} N")
    print(f"  BEMT torque Q_bemt    : {rotor['Q_bemt']:.4f} N·m")
    print(f"  Chord scale factor    : {rotor['scale_c']:.3f}")
    print(f"  DF_scaled range       : {rotor['DF_scaled'].min():.3f} to {rotor['DF_scaled'].max():.3f} (< 0.4)")
    print(f"  RPM                   : {rotor['rpm']:.0f} rpm\n")

    print("Stator (swirl-removal):")
    print(f"  Stator profile drag ≈ {stator['D_profile']:.3f} N")
    
    # NEW PRINT STATEMENTS
    print(f"  Residual Swirl (avg) @ Takeoff : {np.mean(Vtheta_resid_des):.3f} m/s")
    print(f"  Residual Swirl (avg) @ Static  : {np.mean(Vtheta_resid_static):.3f} m/s")
    print(f"  Residual Swirl (avg) @ Cruise  : {np.mean(Vtheta_resid_cruise):.3f} m/s\n")

    rotor_CL = rotor["CL"]
    rotor_CD = cd_from_cl(rotor_CL, cl_opt=rotor["CL_target"], cd_min=0.015, k2=0.04)
    print("Mean aerodynamic coefficients:")
    print(f"  Rotor:  <CL> = {np.mean(rotor_CL):.3f}, <CD> = {np.mean(rotor_CD):.3f}")
    print(f"  Stator: <CL> = {np.mean(stator['CL']):.3f}, <CD> = {np.mean(stator['CD']):.3f}\n")

    print("Annulus geometry (stations 1–4 use same area here):")
    print(f"  R_tip : {R_tip*1000:.1f} mm, R_hub : {R_hub*1000:.1f} mm")
    print(f"  A1 = A2 = A3 = A4 = {A:.6f} m^2\n")

    # --- Design point 1D data ---
    print("=== Design point (TAKEOFF: V_inf = 20 m/s, sea level) ===")
    print(f"  Shaft power P0        : {P0:.1f} W")
    print(f"  Induced velocity v_i  : {v_i_des:.3f} m/s")
    print(f"  Disc speed V2         : {V2_des:.3f} m/s")
    print(f"  Jet speed V4          : {V4_des:.3f} m/s")
    print(f"  Thrust T_design       : {T_des:.3f} N (by construction)")
    print(f"  Mass flow mdot_design : {mdot_des:.3f} kg/s\n")

    # --- Off-design: static ---
    print("=== Off-design: STATIC (V_inf = 0 m/s, sea level, same P0) ===")
    print(f"  Thrust T_static       : {T_static:.3f} N")
    print(f"  Mass flow mdot_static : {mdot_static:.3f} kg/s")
    print(f"  Disc speed V2_static  : {V2_static:.3f} m/s")
    print(f"  Jet speed V4_static   : {V4_static:.3f} m/s\n")

    # --- Off-design: cruise ---
    print("=== Off-design: CRUISE (V_inf = 80 m/s, 10,000 ft, same P0) ===")
    print(f"  ρ_cruise              : {rho_cruise:.4f} kg/m^3")
    print(f"  Thrust T_cruise       : {T_cruise:.3f} N")
    print(f"  Mass flow mdot_cruise : {mdot_cruise:.3f} kg/s")
    print(f"  Disc speed V2_cruise  : {V2_cruise:.3f} m/s")
    print(f"  Jet speed V4_cruise   : {V4_cruise:.3f} m/s\n")

    print("Rotor chord & metal angle (root / mid / tip):")
    for label, idx in [("root", 0), ("mid", len(r_R_rot)//2), ("tip", -1)]:
        print(
            f"  {label:>4}: r/R={r_R_rot[idx]:.2f}, "
            f"c={rotor['c'][idx]*1000:.1f} mm, "
            f"β_m ≈ {beta_m_rotor[idx]:.1f}°"
        )

    # ================= PLOTS =================

    plt.figure()
    plt.plot(r_R_rot, rotor["c"], "o-", label="Rotor chord")
    plt.plot(r_R_sta, stator["c"], "s-", label="Stator chord")
    plt.xlabel("r / R_tip")
    plt.ylabel("Chord [m]")
    plt.title("Chord distribution (rotor & stator)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(r_R_rot, beta_m_rotor, "o-", label="Rotor metal angle")
    plt.plot(r_R_sta, beta_m_stator, "s-", label="Stator metal angle")
    plt.xlabel("r / R_tip")
    plt.ylabel("Metal angle β_m [deg]")
    plt.title("Blade twist (rotor & stator)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()
