import math
import numpy as np
import matplotlib.pyplot as plt
import csv

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def cubic_bezier(t, P0, P1, P2, P3):
    return ((1 - t)**3) * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + (t**3) * P3

def alpha_from_CL_compressible(CL_target, slope_per_deg_low_speed, alpha0_deg, Mach):
    if Mach >= 1.0: beta = 0.1 
    else: beta = math.sqrt(1 - Mach**2)
    slope_comp = slope_per_deg_low_speed / beta
    return CL_target / slope_comp + alpha0_deg

def smooth_list(vals, w=0.3):
    vals = np.asarray(vals, float)
    out = vals.copy()
    n = len(vals)
    for i in range(1, n - 1):
        out[i] = (1.0 - w) * vals[i] + 0.5 * w * (vals[i - 1] + vals[i + 1])
    return out

def cd_from_cl_reynolds(cl, Re, Mach, cl_opt=0.7, cd_min_ref=0.015, Re_ref=5e5):
    if Re < 1000: Re = 1000
    re_factor = (Re_ref / Re)**0.2
    cd_min_local = cd_min_ref * re_factor
    k2 = 0.04
    cd_profile = cd_min_local + k2 * (cl - cl_opt)**2
    cd_wave = 0.0
    if Mach > 0.7: cd_wave = 0.1 * (Mach - 0.7)**2
    return cd_profile + cd_wave

def rho_from_alt_m(alt_m):
    h_km = alt_m / 1000.0
    return 0.0037875*h_km**2 - 0.116575*h_km + 1.225

def viscosity_from_temp(alt_m):
    """Sutherland's Law."""
    if alt_m < 11000: T = 288.15 - 0.0065 * alt_m
    else: T = 216.65
    mu_0 = 1.716e-5
    T_0 = 273.15
    S = 110.4
    mu = mu_0 * ((T/T_0)**1.5) * ((T_0 + S) / (T + S))
    return mu, math.sqrt(1.4 * 287.05 * T)

def design_power_from_T_V(T_des, rho, A, V_inf):
    if rho <= 0 or A <= 0 or T_des <= 0: return 0.0, 0.0, V_inf, V_inf, 0.0
    disc = V_inf**2 + 2.0*T_des/(rho*A)
    if disc <= 0.0: return 0.0, 0.0, V_inf, V_inf, 0.0
    v_i = (-V_inf + math.sqrt(disc)) / 2.0
    V2 = V_inf + v_i
    V4 = V_inf + 2.0*v_i
    P  = T_des * V2
    m_dot = rho * A * V2
    return P, v_i, V2, V4, m_dot

# -----------------------------------------------------------
# Physics: Drag Distribution Calculation
# -----------------------------------------------------------

def calculate_distributed_drag(x_arr, r_arr, V_flow, rho, mu):
    """
    Calculates the drag force distribution along a body profile.
    Uses 'Strip Theory' / Integral Boundary Layer estimation.
    
    Args:
        x_arr, r_arr: Coordinates of the body
        V_flow: The velocity of the air flowing over this specific body.
                (e.g., V_inf for Spinner, V_jet for Tailcone)
    """
    drag_per_length = []
    cf_vals = []
    
    # 1. Calculate Arc Length (s) along the surface
    s = 0.0
    s_arr = [0.0]
    
    for i in range(1, len(x_arr)):
        dx = x_arr[i] - x_arr[i-1]
        dr = r_arr[i] - r_arr[i-1]
        ds = math.sqrt(dx**2 + dr**2)
        s += ds
        s_arr.append(s)
        
    # 2. Calculate Local Drag at each station
    for i in range(len(x_arr)):
        if i == 0: 
            drag_per_length.append(0.0)
            cf_vals.append(0.0)
            continue
            
        # Local Reynolds Number based on running length s
        Re_s = (rho * V_flow * s_arr[i]) / mu
        if Re_s < 1.0: Re_s = 1.0
        
        # Skin Friction Coefficient (C_f)
        if Re_s < 5e5:
            # Laminar (Blasius)
            Cf = 0.664 / math.sqrt(Re_s)
        else:
            # Turbulent (Prandtl-Schlichting)
            Cf = 0.455 / (math.log10(Re_s)**2.58)
            
        cf_vals.append(Cf)
        
        # Drag Force on this segment (approximate strip)
        perimeter = 2.0 * math.pi * r_arr[i]
        tau_w = 0.5 * rho * (V_flow**2) * Cf
        
        dx_local = x_arr[i] - x_arr[i-1]
        if dx_local < 1e-9: dx_local = 1e-9
        
        dr_local = r_arr[i] - r_arr[i-1]
        ds_local = math.sqrt(dx_local**2 + dr_local**2)
        
        # Project shear stress into axial direction
        cos_theta = dx_local / ds_local
        
        # Local Drag Force
        d_force = tau_w * (perimeter * ds_local) * cos_theta
        
        # Normalize to Drag per meter (dD/dx)
        drag_density = d_force / dx_local 
        drag_per_length.append(drag_density)
        
    # FIX: Return numpy arrays to ensure mathematical operations (like *1000) work correctly
    return np.array(drag_per_length), np.array(cf_vals), np.array(s_arr)

# -----------------------------------------------------------
# Geometry Generators
# -----------------------------------------------------------

def generate_bellmouth_profile(D_duct, D_inlet, L, n_points=60):
    R_throat = 0.5 * D_duct
    R_mouth  = 0.5 * D_inlet
    x = np.linspace(0.0, L, n_points)
    if L < 1e-9: return x, np.full_like(x, R_throat)
    t = x / L
    f = 3.0 * t**2 - 2.0 * t**3
    r = R_throat + (R_mouth - R_throat) * f
    return x, r 

def generate_spinner_profile(D_hub, L_over_D=1.0, n_points=60):
    R_hub = 0.5 * D_hub
    L = L_over_D * D_hub 
    x = np.linspace(-L, 0, n_points)
    r = R_hub * np.sqrt(1 - (x/L)**2)
    return x, r

def generate_tailcone_profile(D_hub, L_over_D=2.5, n_points=60):
    R_hub = 0.5 * D_hub
    L = L_over_D * D_hub
    x = np.linspace(0, L, n_points)
    r = R_hub * (1 - (x/L)**2)
    r = np.maximum(r, 0)
    return x, r

def generate_nacelle_oml(D_max, L_total, D_exit, D_inlet, x_max_loc=0.35, n_points=100):
    x = np.linspace(0, L_total, n_points)
    R_max = D_max / 2.0
    R_inlet = D_inlet / 2.0
    R_exit = D_exit / 2.0
    r = np.zeros_like(x)
    L_fore = L_total * x_max_loc
    L_aft = L_total - L_fore
    for i, xi in enumerate(x):
        if xi <= L_fore:
            t = xi / L_fore
            r[i] = R_inlet + (R_max - R_inlet) * (1 - (1-t)**3)**(1/3) 
            if r[i] > R_max: r[i] = R_max
        else:
            t = (xi - L_fore) / L_aft
            r[i] = cubic_bezier(t, R_max, R_max, (R_max+R_exit)/2, R_exit)
    return x, r

def generate_nozzle_iml(D_duct, D_exit, L_nozzle, n_points=40):
    R_duct = D_duct / 2.0
    R_exit = D_exit / 2.0
    x = np.linspace(0, L_nozzle, n_points)
    t = x / L_nozzle
    r = (1-t)**3 * R_duct + 3*(1-t)**2 * t * R_duct + 3*(1-t)*t**2 * R_exit + t**3 * R_exit
    return x, r

# -----------------------------------------------------------
# Rotor design functions
# -----------------------------------------------------------

def design_rotor_base(
    D=0.125, T_des=25.0, phi_mean=0.75, DF_limit=0.4, hub_to_tip=0.30, 
    rho=1.225, psi_des=0.4, B_rotor=11, n_span=16, cR_min=0.04, cR_max=0.30, 
    CL_target=0.9, alt_m=0.0
):
    mu, a_sound = viscosity_from_temp(alt_m)
    R_tip = 0.5 * D
    R_hub = hub_to_tip * R_tip
    A_annulus = math.pi * (R_tip**2 - R_hub**2)
    V4 = math.sqrt(2.0 * T_des / (rho * A_annulus))
    dh0 = 0.5 * V4**2
    U_m = math.sqrt(dh0 / psi_des)
    r_m = math.sqrt(0.5 * (R_hub**2 + R_tip**2))
    omega = U_m / r_m
    rpm = omega * 60.0 / (2.0 * math.pi)
    Vx_const = phi_mean * U_m
    r = np.linspace(R_hub * 1.03, R_tip * 0.99, n_span)
    r_R = r / R_tip
    c = np.zeros_like(r)
    beta1 = np.zeros_like(r)
    alpha = np.zeros_like(r)
    CL_arr = np.zeros_like(r)
    CD_arr = np.zeros_like(r)
    Re_arr = np.zeros_like(r)
    Mach_arr = np.zeros_like(r)
    TipLoss_F = np.zeros_like(r)
    cR_root = 0.18
    cR_tip  = 0.06
    cR1     = 0.16
    cR2     = 0.10
    
    for i, ri in enumerate(r):
        Ui = omega * ri
        Vtheta2i = dh0 / Ui 
        Vtheta1i = 0.0      
        W1_sq = Vx_const**2 + (Ui - Vtheta1i)**2
        W1 = math.sqrt(W1_sq)
        W2_sq = Vx_const**2 + (Ui - Vtheta2i)**2
        W2 = math.sqrt(W2_sq)
        phi_flow = math.atan2(Vx_const, Ui - Vtheta2i) 
        f_exp = -(B_rotor/2.0) * (R_tip - ri) / (R_tip * math.sin(phi_flow))
        if f_exp > 0: f_exp = 0
        if f_exp < -20: f_exp = -20
        F = (2.0/math.pi) * math.acos(math.exp(f_exp))
        if F < 0.1: F = 0.1 
        TipLoss_F[i] = F
        ti = (ri - R_hub) / (R_tip - R_hub)
        c_guess_R = cubic_bezier(ti, cR_root, cR1, cR2, cR_tip)
        c_guess = c_guess_R * R_tip
        Re = (rho * W1 * c_guess) / mu
        Mach = W1 / a_sound
        Re_arr[i] = Re
        Mach_arr[i] = Mach
        a_low_speed = 0.11 
        alpha_req = alpha_from_CL_compressible(CL_target, a_low_speed, -2.0, Mach)
        CD_val = cd_from_cl_reynolds(CL_target, Re, Mach, cl_opt=0.7)
        beta1_deg = math.degrees(math.atan2(Vx_const, Ui - Vtheta1i))
        s = 2.0 * math.pi * ri / B_rotor
        dVtheta = Vtheta2i - Vtheta1i
        denom_df = 2.0 * W1 * (DF_limit - 1.0 + W2/W1)
        c_min_df = 0.0
        if denom_df > 1e-9 and dVtheta > 0:
             c_min_df = (dVtheta * s) / denom_df
        c_final = max(c_guess, c_min_df)
        c_final = min(c_final, cR_max * R_tip)
        c_final = max(c_final, cR_min * R_tip)
        if ri > 0.98 * R_tip:
            CD_val *= 1.5
        c[i] = c_final
        beta1[i] = beta1_deg
        alpha[i] = alpha_req
        CL_arr[i] = CL_target
        CD_arr[i] = CD_val

    c = smooth_list(c, w=0.3)
    rho_mat = 1150.0 
    CF_force = 0.0
    dr = (R_tip - R_hub) / (n_span - 1)
    for i in range(len(r)):
        Area_sect = c[i] * (0.12 * c[i]) 
        dF = rho_mat * Area_sect * (omega**2) * r[i] * dr
        CF_force += dF
    Root_Area = c[0] * (0.12 * c[0])
    Root_Stress_Pa = CF_force / Root_Area if Root_Area > 0 else 0
    Root_Stress_MPa = Root_Stress_Pa / 1e6
    V_disc_eff = 0.5 * V4
    T_mom = 0.5 * rho * A_annulus * V4**2

    return {
        "r": r, "r_R": r_R, "c": c, "beta1": beta1, "rpm": rpm, 
        "R_tip": R_tip, "R_hub": R_hub, "A_annulus": A_annulus, "V4": V4,
        "V_disc_eff": V_disc_eff, "T_mom": T_mom, 
        "CL": CL_arr, "alpha": alpha, "CD": CD_arr,
        "Re": Re_arr, "Mach": Mach_arr, "TipLoss_F": TipLoss_F,
        "Stress_MPa": Root_Stress_MPa, "CF_Force": CF_force
    }


# -----------------------------------------------------------
# Full Nacelle & Inlet Design
# -----------------------------------------------------------

def design_nacelle_system(
    D_duct=0.125, T_des=25.0, V_inf=20.0, alt_m=0.0, hub_ratio=0.3, n_points=60
):
    # 1. Ambient & Fan Performance
    rho = rho_from_alt_m(alt_m)
    mu, _ = viscosity_from_temp(alt_m)
    A_duct = math.pi * (D_duct**2) / 4.0
    P0, v_i, V2, V4, m_dot = design_power_from_T_V(T_des, rho, A_duct, V_inf)

    # 2. Inlet Optimization
    best_penalty = 1e9
    best_params = (1.3, 0.6) 
    cr_sweep = np.linspace(1.15, 1.5, 8)
    ld_sweep = np.linspace(0.4, 0.8, 8)

    for cr_val in cr_sweep:
        for ld_val in ld_sweep:
            D_inlet_local = math.sqrt(cr_val) * D_duct
            L_local = ld_val * D_inlet_local
            surface_area = math.pi * (D_duct/2 + D_inlet_local/2) * math.sqrt(L_local**2 + (D_inlet_local/2 - D_duct/2)**2)
            drag_friction = 0.5 * rho * (V_inf**2) * surface_area * 0.003
            loss_factor = 0.01 / (ld_val**2 + 0.001) + 0.005 / ((cr_val-1.0)**2 + 0.01)
            q_duct = 0.5 * rho * V2**2
            drag_separation_penalty = q_duct * A_duct * loss_factor
            total_penalty = drag_friction + drag_separation_penalty
            if total_penalty < best_penalty:
                best_penalty = total_penalty
                best_params = (cr_val, ld_val)

    CR_inlet, L_inlet_over_D = best_params
    D_inlet = math.sqrt(CR_inlet) * D_duct
    L_inlet = L_inlet_over_D * D_inlet
    
    # 3. Nozzle Sizing
    if V4 > 0: A_exit = m_dot / (rho * V4)
    else: A_exit = A_duct 
    D_exit_nozzle = math.sqrt(4.0 * A_exit / math.pi)
    Nozzle_CR = A_duct / A_exit
    
    # 4. Length Definitions
    L_fan_section = 0.0353 
    L_nozzle = 1.0 * D_duct 
    D_hub = D_duct * hub_ratio
    L_tailcone = 2.5 * D_hub 
    
    # 5. Generate Component Profiles
    x_spin, r_spin = generate_spinner_profile(D_hub, L_over_D=1.0, n_points=n_points)
    x_bell_gen, r_bell_gen = generate_bellmouth_profile(D_duct, D_inlet, L_inlet, n_points)
    x_inlet_inner = -x_bell_gen[::-1] 
    r_inlet_inner = r_bell_gen[::-1]
    x_fan = np.linspace(0, L_fan_section, int(n_points/2))
    r_fan = np.full_like(x_fan, D_duct/2.0)
    x_nozz_gen, r_nozz_gen = generate_nozzle_iml(D_duct, D_exit_nozzle, L_nozzle, n_points)
    x_nozzle = x_nozz_gen + L_fan_section
    r_nozzle = r_nozz_gen
    x_tail_gen, r_tail_gen = generate_tailcone_profile(D_hub, L_over_D=2.5, n_points=n_points)
    x_tail = x_tail_gen + L_fan_section
    r_tail = r_tail_gen
    
    # F. Nacelle Outer Mold Line (OML)
    Total_Axial_Length = L_inlet + L_fan_section + L_nozzle
    D_highlight = D_inlet * 1.02 
    D_min_physical = D_duct + 2.0 * 0.004 
    D_max_nacelle = max(D_min_physical, D_highlight * 1.05)
    D_exit_oml = D_exit_nozzle + 2.0 * 0.002 
    
    x_oml_gen, r_oml_gen = generate_nacelle_oml(D_max_nacelle, Total_Axial_Length, D_exit_oml, D_highlight, x_max_loc=0.35, n_points=150)
    x_oml = x_oml_gen - L_inlet
    r_oml = r_oml_gen
    
    # 6. Calculate Drag Distribution on COMPONENTS
    
    # A. OML (Outer Skin) - Exposed to Freestream V_inf
    oml_drag_dens, oml_cf, oml_s = calculate_distributed_drag(x_oml, r_oml, V_inf, rho, mu)
    
    # B. Spinner (Nose Cone) - Exposed to Freestream V_inf (approx)
    spin_drag_dens, spin_cf, spin_s = calculate_distributed_drag(x_spin, r_spin, V_inf, rho, mu)
    
    # C. Tail Cone (Afterbody) - Exposed to JET Velocity V4
    tail_drag_dens, tail_cf, tail_s = calculate_distributed_drag(x_tail, r_tail, V4, rho, mu)

    return {
        "D_duct": D_duct, "D_exit_nozzle": D_exit_nozzle, "V4": V4,
        "x_spin": x_spin, "r_spin": r_spin,
        "x_inlet": x_inlet_inner, "r_inlet": r_inlet_inner,
        "x_fan": x_fan, "r_fan": r_fan,
        "x_nozzle": x_nozzle, "r_nozzle": r_nozzle,
        "x_tail": x_tail, "r_tail": r_tail,
        "x_oml": x_oml, "r_oml": r_oml,
        "Nozzle_CR": Nozzle_CR, "T_des": T_des,
        "CR_opt": CR_inlet, "L_over_D_opt": L_inlet_over_D,
        
        # Drag Data
        "oml_drag": oml_drag_dens, "oml_cf": oml_cf, "oml_s": oml_s,
        "spin_drag": spin_drag_dens, "spin_cf": spin_cf, "spin_s": spin_s,
        "tail_drag": tail_drag_dens, "tail_cf": tail_cf, "tail_s": tail_s
    }

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":
    print("--- Designing EDF Nacelle System ---")
    
    D_fan = 0.125
    Thrust = 25.0
    V_flight = 20.0
    
    # 1. Run Rotor Design with Physics
    rotor = design_rotor_base(D=D_fan, T_des=Thrust, DF_limit=0.45, CL_target=0.9, alt_m=0.0)
    
    # 2. Run Nacelle Design
    sys = design_nacelle_system(D_duct=D_fan, T_des=Thrust, V_inf=V_flight, alt_m=0.0)
    
    print(f"\nDrag Analysis (Approx. Skin Friction):")
    
    drag_oml = np.trapz(sys['oml_drag'], sys['x_oml'])
    drag_spin = np.trapz(sys['spin_drag'], sys['x_spin'])
    drag_tail = np.trapz(sys['tail_drag'], sys['x_tail'])
    
    print(f"  OML Drag      : {drag_oml:.4f} N (Velocity ~ {V_flight:.1f} m/s)")
    print(f"  Spinner Drag  : {drag_spin:.4f} N (Velocity ~ {V_flight:.1f} m/s)")
    print(f"  Tail Cone Drag: {drag_tail:.4f} N (Velocity ~ {sys['V4']:.1f} m/s)")

    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Nacelle Geometry
    plt.subplot(3,1,1)
    plt.plot(sys['x_oml']*1000, sys['r_oml']*1000, 'k-', linewidth=2.5, label='Nacelle OML')
    plt.plot(sys['x_oml']*1000, -sys['r_oml']*1000, 'k-', linewidth=2.5)
    x_inner = np.concatenate([sys['x_inlet'], sys['x_fan'], sys['x_nozzle']])
    r_inner = np.concatenate([sys['r_inlet'], sys['r_fan'], sys['r_nozzle']])
    plt.plot(x_inner*1000, r_inner*1000, 'b-', linewidth=1.5, label='Inner Duct Wall')
    plt.plot(x_inner*1000, -r_inner*1000, 'b-', linewidth=1.5)
    plt.plot(sys['x_spin']*1000, sys['r_spin']*1000, 'r-', label='Centerbody')
    plt.plot(sys['x_spin']*1000, -sys['r_spin']*1000, 'r-')
    plt.plot(sys['x_tail']*1000, sys['r_tail']*1000, 'r-')
    plt.plot(sys['x_tail']*1000, -sys['r_tail']*1000, 'r-')
    plt.title("Optimized EDF Nacelle Geometry")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Drag Distribution
    plt.subplot(3,1,2)
    plt.plot(sys['x_oml']*1000, sys['oml_drag'], 'r-', label='OML Drag (dD/dx)')
    plt.plot(sys['x_spin']*1000, sys['spin_drag'], 'g-', label='Spinner Drag')
    plt.plot(sys['x_tail']*1000, sys['tail_drag'], 'b-', label='Tail Cone Drag')
    
    plt.fill_between(sys['x_oml']*1000, sys['oml_drag'], color='red', alpha=0.1)
    plt.fill_between(sys['x_spin']*1000, sys['spin_drag'], color='green', alpha=0.1)
    plt.fill_between(sys['x_tail']*1000, sys['tail_drag'], color='blue', alpha=0.1)
    
    plt.title('Drag Force Distribution (Force per unit length)')
    plt.ylabel('Drag Force [N/m]')
    plt.xlabel('Axial Position [mm]')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Skin Friction Coefficient
    plt.subplot(3,1,3)
    plt.plot(sys['oml_s']*1000, sys['oml_cf'], 'r--', label='OML Cf')
    plt.plot(sys['spin_s']*1000, sys['spin_cf'], 'g--', label='Spinner Cf')
    plt.plot(sys['tail_s']*1000, sys['tail_cf'], 'b--', label='Tail Cf')
    plt.title('Local Skin Friction Coefficient vs Running Length')
    plt.xlabel('Surface Arc Length s [mm]')
    plt.ylabel('Cf')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()