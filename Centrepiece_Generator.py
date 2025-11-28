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
    """Linear lift curve: CL = a * (alpha - alpha0)."""
    return CL_target / slope_per_deg + alpha0_deg


def smooth_list(vals, w=0.3):
    """Light smoothing of a 1D array."""
    vals = np.asarray(vals, float)
    out = vals.copy()
    n = len(vals)
    for i in range(1, n - 1):
        out[i] = (1.0 - w) * vals[i] + 0.5 * w * (vals[i - 1] + vals[i + 1])
    return out


def cd_from_cl(cl, cl_opt=0.7, cd_min=0.015, k2=0.04):
    """Simple parabolic drag polar."""
    cl = np.asarray(cl, float)
    return cd_min + k2 * (cl - cl_opt)**2


def rho_from_alt_m(alt_m):
    """Density interpolation from altitude."""
    h_km = alt_m / 1000.0
    return 0.0037875*h_km**2 - 0.116575*h_km + 1.225


def design_power_from_T_V(T_des, rho, A, V_inf):
    """
    Simple actuator-disc relations.
    Returns P, v_i, V2, V4, m_dot.
    """
    if rho <= 0 or A <= 0 or T_des <= 0:
        return 0.0, 0.0, V_inf, V_inf, 0.0

    disc = V_inf**2 + 2.0*T_des/(rho*A)
    if disc <= 0.0:
        return 0.0, 0.0, V_inf, V_inf, 0.0

    v_i = (-V_inf + math.sqrt(disc)) / 2.0
    V2 = V_inf + v_i
    V4 = V_inf + 2.0*v_i
    P  = T_des * V2
    m_dot = rho * A * V2
    return P, v_i, V2, V4, m_dot


# -----------------------------------------------------------
# Geometry Generators
# -----------------------------------------------------------

def generate_bellmouth_profile(D_duct, D_inlet, L, n_points=60):
    """Outer bellmouth profile (C1 cubic)."""
    R_throat = 0.5 * D_duct
    R_mouth  = 0.5 * D_inlet
    x = np.linspace(0.0, L, n_points)
    if L < 1e-9: return x, np.full_like(x, R_throat)
    t = x / L
    f = 3.0 * t**2 - 2.0 * t**3
    r = R_throat + (R_mouth - R_throat) * f
    return x, r 


def generate_spinner_profile(D_hub, L_over_D=1.0, n_points=60):
    """Elliptical nose cone."""
    R_hub = 0.5 * D_hub
    L = L_over_D * D_hub 
    x = np.linspace(-L, 0, n_points)
    r = R_hub * np.sqrt(1 - (x/L)**2)
    return x, r


def generate_tailcone_profile(D_hub, L_over_D=2.5, n_points=60):
    """
    Generate an aerodynamic tail cone (afterbody) profile.
    Uses a parabolic/elliptical closing shape to minimize base drag.
    Literature: Hoerner, 'Fluid-Dynamic Drag'.
    """
    R_hub = 0.5 * D_hub
    L = L_over_D * D_hub
    x = np.linspace(0, L, n_points)
    r = R_hub * (1 - (x/L)**2)
    r = np.maximum(r, 0)
    return x, r


def generate_nacelle_oml(D_max, L_total, D_exit, D_inlet, x_max_loc=0.35, n_points=100):
    """
    Generate Nacelle Outer Mold Line (OML) using a NACA 6-series like distribution.
    Adapted for a pod shape.
    """
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
            # Standard NACA forebody approx (Power law shape)
            r[i] = R_inlet + (R_max - R_inlet) * (1 - (1-t)**3)**(1/3) 
            if r[i] > R_max: r[i] = R_max
        else:
            # Afterbody: Taper from R_max to R_exit
            t = (xi - L_fore) / L_aft
            # Cubic taper
            r[i] = cubic_bezier(t, R_max, R_max, (R_max+R_exit)/2, R_exit)
            
    return x, r

def generate_nozzle_iml(D_duct, D_exit, L_nozzle, n_points=40):
    """Inner flowpath for the nozzle (Duct to Exit)."""
    R_duct = D_duct / 2.0
    R_exit = D_exit / 2.0
    x = np.linspace(0, L_nozzle, n_points)
    
    t = x / L_nozzle
    # Smooth cubic transition
    r = (1-t)**3 * R_duct + 3*(1-t)**2 * t * R_duct + 3*(1-t)*t**2 * R_exit + t**3 * R_exit
    return x, r


# -----------------------------------------------------------
# Rotor design functions (kept for consistency)
# -----------------------------------------------------------

def design_rotor_base(D=0.125, T_des=25.0, phi_mean=0.75, DF_limit=0.4, hub_to_tip=0.30, rho=1.225, psi_des=0.4, B_rotor=11, n_span=16, cR_min=0.04, cR_max=0.30, CL_target=0.9):
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
    U = omega * r
    Vx = np.full_like(r, Vx_const)
    Vtheta2 = dh0 / U
    V1 = np.sqrt(Vx**2)
    V2 = np.sqrt(Vx**2 + Vtheta2**2)
    beta1 = np.degrees(np.arctan2(Vx, U))
    alpha_section = alpha_from_CL(CL_target, 0.11, -2.0)
    CL = np.full_like(r, CL_target)
    alpha = np.full_like(r, alpha_section)
    
    c_pref = np.full_like(r, 0.02)
    c = c_pref
    DF = np.zeros_like(r)
    
    V_disc_eff = 0.5 * V4
    T_mom = 0.5 * rho * A_annulus * V4**2
    CD = cd_from_cl(CL)

    return {
        "r": r, "r_R": r_R, "c": c, "beta1": beta1, "DF": DF, "rpm": rpm, 
        "R_tip": R_tip, "R_hub": R_hub, "A_annulus": A_annulus, "V4": V4,
        "V_disc_eff": V_disc_eff, "T_mom": T_mom, "CL": CL, "alpha": alpha, "CD": CD
    }


# -----------------------------------------------------------
# Full Nacelle & Inlet Design
# -----------------------------------------------------------

def design_nacelle_system(
    D_duct=0.125,       # [m] duct / rotor-tip diameter
    T_des=25.0,         # [N] design thrust at takeoff
    V_inf=20.0,         # [m/s] design flight speed
    alt_m=0.0,          # [m] design altitude
    hub_ratio=0.3,      # Hub ratio
    n_points=60
):
    """
    Design the complete EDF Aerodynamic Flowpath and Nacelle.
    Includes: Spinner, Bellmouth, Duct, Nozzle, Tailcone, and Outer Mold Line.
    
    Optimization:
    - Minimizes Drag (Friction + Separation) for Inlet.
    - Matches Nozzle area for momentum thrust.
    - Sizes Nacelle OML for minimal frontal area while maintaining structural enclosure.
    """

    # 1. Ambient & Fan Performance
    rho = rho_from_alt_m(alt_m)
    A_duct = math.pi * (D_duct**2) / 4.0
    P0, v_i, V2, V4, m_dot = design_power_from_T_V(T_des, rho, A_duct, V_inf)

    # 2. Inlet Optimization Loop (Seddon & Goldsmith)
    # We sweep CR and L/D to find minimum penalty (Drag + Separation Risk).
    
    best_penalty = 1e9
    best_params = (1.3, 0.6) # Fallback

    # Optimization ranges:
    # CR: 1.15 to 1.5 (Standard for Bellmouths)
    # L/D: 0.4 to 0.8 (Short vs Long intake trade-off)
    cr_sweep = np.linspace(1.15, 1.5, 8)
    ld_sweep = np.linspace(0.4, 0.8, 8)

    for cr_val in cr_sweep:
        for ld_val in ld_sweep:
            D_inlet_local = math.sqrt(cr_val) * D_duct
            L_local = ld_val * D_inlet_local
            
            # Approx Wetted Area (Conical Frustum)
            surface_area = math.pi * (D_duct/2 + D_inlet_local/2) * math.sqrt(L_local**2 + (D_inlet_local/2 - D_duct/2)**2)
            
            # Drag Component 1: Skin Friction (Favor Small L/D, Small CR)
            # C_f approx 0.003
            drag_friction = 0.5 * rho * (V_inf**2) * surface_area * 0.003

            # Drag Component 2: Lip Separation Risk (Favor Large L/D, Large CR)
            # Loss coeff xi rises sharply if L/D < 0.2 or CR < 1.1
            # Model: xi = k / (L/D)^2 + k2 / (CR-1)
            loss_factor = 0.01 / (ld_val**2 + 0.001) + 0.005 / ((cr_val-1.0)**2 + 0.01)
            
            # Penalty is weighted sum
            q_duct = 0.5 * rho * V2**2
            drag_separation_penalty = q_duct * A_duct * loss_factor
            
            total_penalty = drag_friction + drag_separation_penalty
            
            if total_penalty < best_penalty:
                best_penalty = total_penalty
                best_params = (cr_val, ld_val)

    # Apply Optimal Parameters
    CR_inlet, L_inlet_over_D = best_params
    
    D_inlet = math.sqrt(CR_inlet) * D_duct
    L_inlet = L_inlet_over_D * D_inlet
    
    # 3. Nozzle Sizing
    # Calculate required exit area to achieve Jet Velocity V4
    if V4 > 0:
        A_exit = m_dot / (rho * V4)
    else:
        A_exit = A_duct 
        
    D_exit_nozzle = math.sqrt(4.0 * A_exit / math.pi)
    Nozzle_CR = A_duct / A_exit
    
    # 4. Length Definitions
    # User constraint: Fan Section Length = 3.53 cm
    L_fan_section = 0.0353 
    
    # Nozzle length: Typically 1.0 * D_duct for smooth transition
    L_nozzle = 1.0 * D_duct 
    
    # Tail Cone length: Fineness ratio ~2.5 behind hub
    D_hub = D_duct * hub_ratio
    L_tailcone = 2.5 * D_hub 
    
    # 5. Generate Component Profiles
    
    # A. Spinner (Nose Cone)
    # Ends at x=0
    x_spin, r_spin = generate_spinner_profile(D_hub, L_over_D=1.0, n_points=n_points)
    
    # B. Inlet Bellmouth (Inner)
    # Starts at x = -L_inlet, ends at x = 0
    x_bell_gen, r_bell_gen = generate_bellmouth_profile(D_duct, D_inlet, L_inlet, n_points)
    x_inlet_inner = -x_bell_gen[::-1] 
    r_inlet_inner = r_bell_gen[::-1]
    
    # C. Fan Shroud (Inner)
    # x=0 to x=L_fan_section. STRICTLY D_duct.
    x_fan = np.linspace(0, L_fan_section, int(n_points/2))
    r_fan = np.full_like(x_fan, D_duct/2.0)
    
    # D. Nozzle (Inner)
    x_nozz_gen, r_nozz_gen = generate_nozzle_iml(D_duct, D_exit_nozzle, L_nozzle, n_points)
    x_nozzle = x_nozz_gen + L_fan_section
    r_nozzle = r_nozz_gen
    
    # E. Tail Cone (Centerbody)
    # Starts at L_fan_section
    x_tail_gen, r_tail_gen = generate_tailcone_profile(D_hub, L_over_D=2.5, n_points=n_points)
    x_tail = x_tail_gen + L_fan_section
    r_tail = r_tail_gen
    
    # F. Nacelle Outer Mold Line (OML)
    # Total Length
    Total_Axial_Length = L_inlet + L_fan_section + L_nozzle
    
    # OML Sizing
    # 1. Inlet Lip: Must match D_inlet outer edge.
    #    Bellmouth usually has a lip radius. We assume D_inlet is the flow highlight.
    #    The OML Highlight usually equals D_inlet or slightly larger (blunt lip).
    D_highlight = D_inlet * 1.02 # 2% lip thickness factor
    
    # 2. Max Diameter:
    #    Must clear the internal duct D_duct.
    #    Must provide wall thickness for structure/motor.
    #    Wall thickness t_wall ~ 2-3mm minimum + structure.
    #    Let's assume min physical wall thickness of 3mm (0.003m) on radius.
    D_min_physical = D_duct + 2.0 * 0.004 # 4mm wall thickness
    
    #    We also need aerodynamic curvature.
    #    NACA 1-series cowlings often have D_max ~ D_highlight + curvature.
    #    Let's optimize D_max to be just enough to cover D_min_physical and provide smooth curve from D_highlight.
    D_max_nacelle = max(D_min_physical, D_highlight * 1.05)
    
    # 3. Exit Diameter
    #    Must clear Nozzle Exit + Wall Thickness.
    D_exit_oml = D_exit_nozzle + 2.0 * 0.002 # 2mm wall at trailing edge
    
    x_oml_gen, r_oml_gen = generate_nacelle_oml(
        D_max=D_max_nacelle, 
        L_total=Total_Axial_Length, 
        D_exit=D_exit_oml, 
        D_inlet=D_highlight, 
        x_max_loc=0.35, 
        n_points=150
    )
    # Shift OML x to start at -L_inlet
    x_oml = x_oml_gen - L_inlet
    r_oml = r_oml_gen

    return {
        "D_duct": D_duct, "D_exit_nozzle": D_exit_nozzle, "V4": V4,
        "x_spin": x_spin, "r_spin": r_spin,
        "x_inlet": x_inlet_inner, "r_inlet": r_inlet_inner,
        "x_fan": x_fan, "r_fan": r_fan,
        "x_nozzle": x_nozzle, "r_nozzle": r_nozzle,
        "x_tail": x_tail, "r_tail": r_tail,
        "x_oml": x_oml, "r_oml": r_oml,
        "Nozzle_CR": Nozzle_CR, "T_des": T_des,
        "CR_opt": CR_inlet, "L_over_D_opt": L_inlet_over_D
    }

# -----------------------------------------------------------
# Save output distributions to CSV (Stub for compatibility)
# -----------------------------------------------------------
def save_to_csv(rotor, file_name="rotor_distribution.csv"):
    pass 

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":
    print("--- Designing EDF Nacelle System ---")
    
    # Inputs
    D_fan = 0.125
    Thrust = 25.0
    V_flight = 20.0
    
    # Run Design
    sys = design_nacelle_system(D_duct=D_fan, T_des=Thrust, V_inf=V_flight, alt_m=0.0)
    
    print(f"Design Result:")
    print(f"  Thrust Target : {sys['T_des']:.1f} N")
    print(f"  Duct Inner Dia: {sys['D_duct']*1000:.1f} mm")
    print(f"  Jet Velocity  : {sys['V4']:.1f} m/s")
    print(f"  Nozzle CR     : {sys['Nozzle_CR']:.3f}")
    print(f"  Inlet Opt CR  : {sys['CR_opt']:.3f}")
    print(f"  Inlet Opt L/D : {sys['L_over_D_opt']:.3f}")

    # Plotting
    plt.figure(figsize=(14, 6))
    
    # 1. Outer Mold Line (Nacelle Skin)
    plt.plot(sys['x_oml']*1000, sys['r_oml']*1000, 'k-', linewidth=2.5, label='Nacelle OML')
    plt.plot(sys['x_oml']*1000, -sys['r_oml']*1000, 'k-', linewidth=2.5)
    
    # 2. Inner Flowpath (Inlet -> Fan -> Nozzle)
    x_inner = np.concatenate([sys['x_inlet'], sys['x_fan'], sys['x_nozzle']])
    r_inner = np.concatenate([sys['r_inlet'], sys['r_fan'], sys['r_nozzle']])
    
    plt.plot(x_inner*1000, r_inner*1000, 'b-', linewidth=1.5, label='Inner Duct Wall')
    plt.plot(x_inner*1000, -r_inner*1000, 'b-', linewidth=1.5)
    
    # 3. Centerbody (Spinner -> Hub -> Tailcone)
    x_hub = sys['x_fan']
    r_hub = np.full_like(x_hub, sys['r_spin'][-1])
    
    plt.plot(sys['x_spin']*1000, sys['r_spin']*1000, 'r-', label='Centerbody')
    plt.plot(sys['x_spin']*1000, -sys['r_spin']*1000, 'r-')
    plt.plot(x_hub*1000, r_hub*1000, 'r--')
    plt.plot(x_hub*1000, -r_hub*1000, 'r--')
    plt.plot(sys['x_tail']*1000, sys['r_tail']*1000, 'r-')
    plt.plot(sys['x_tail']*1000, -sys['r_tail']*1000, 'r-')
    
    # Fill styling
    plt.fill_between(sys['x_oml']*1000, sys['r_oml']*1000, 150, color='gray', alpha=0.1)
    plt.fill_between(sys['x_oml']*1000, -sys['r_oml']*1000, -150, color='gray', alpha=0.1)
    
    plt.title("Optimized EDF Nacelle Geometry (Inlet, Duct, Nozzle, Tailcone)")
    plt.xlabel("Axial Position [mm] (0 = Fan Face)")
    plt.ylabel("Radius [mm]")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
