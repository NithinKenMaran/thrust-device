import math
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# PHYSICS ENGINE: BOUNDARY LAYER DRAG
# ==============================================================================

def get_atmosphere(alt_m=0.0):
    """Standard Atmosphere Properties."""
    if alt_m < 11000:
        T = 288.15 - 0.0065 * alt_m
        rho = 1.225 * (T / 288.15)**4.2558
    else:
        T = 216.65
        rho = 0.3639 * math.exp(-(alt_m - 11000) / 6340)
    
    # Sutherland's Viscosity
    mu_0 = 1.716e-5
    T_0 = 273.15
    S = 110.4
    mu = mu_0 * ((T/T_0)**1.5) * ((T_0 + S) / (T + S))
    return rho, mu

def calculate_jet_velocity(Thrust, V_inf, D_fan, rho):
    """
    Calculates the Exhaust Jet Velocity (V4) required to produce the Thrust
    at a given flight speed V_inf using Actuator Disc Theory.
    T = m_dot * (V4 - V_inf)
    """
    A = math.pi * (D_fan/2)**2
    
    # Quadratic solution for induced velocity v_i
    # T = 2 * rho * A * (V_inf + v_i) * v_i
    # 2*rho*A * vi^2 + 2*rho*A*V_inf * vi - T = 0
    
    a = 2 * rho * A
    b = 2 * rho * A * V_inf
    c = -Thrust
    
    delta = b**2 - 4*a*c
    if delta < 0: return V_inf # Should not happen for positive thrust
    
    v_i = (-b + math.sqrt(delta)) / (2*a)
    V_jet = V_inf + 2*v_i
    return V_jet

def calculate_component_drag(x_arr, r_arr, V_flow, rho, mu):
    """
    Calculates distributed skin friction drag along a profile.
    Returns: x, Drag_Density (N/m), Cumulative_Drag (N)
    """
    x_arr = np.array(x_arr)
    r_arr = np.array(r_arr)
    
    drag_dist = []
    total_drag = 0.0
    s_accum = 0.0
    
    for i in range(1, len(x_arr)):
        # Geometry of segment
        dx = x_arr[i] - x_arr[i-1]
        dr = r_arr[i] - r_arr[i-1]
        ds = math.sqrt(dx**2 + dr**2)
        r_avg = (r_arr[i] + r_arr[i-1]) / 2.0
        
        s_accum += ds
        
        # Local Reynolds No (Running length)
        Re_x = (rho * V_flow * s_accum) / mu
        if Re_x < 100: Re_x = 100
        
        # Skin Friction Coeff (Mixed Laminar/Turbulent)
        # Transition approx at 5e5
        if Re_x < 5e5:
            Cf = 0.664 / math.sqrt(Re_x)
        else:
            # Turbulent Prandtl-Schlichting
            Cf = 0.455 / (math.log10(Re_x)**2.58)
            
        # Wall Shear Stress
        tau = 0.5 * rho * V_flow**2 * Cf
        
        # Drag Force on Segment (Projected to Axial direction)
        # Area = 2 * pi * r * ds
        # Axial Component = Force * (dx/ds)
        
        d_force = tau * (2 * math.pi * r_avg * ds) * (dx/ds)
        
        if dx > 1e-9:
            drag_dist.append(d_force / dx)
        else:
            drag_dist.append(0)
            
        total_drag += d_force
        
    # Pad first element
    drag_dist.insert(0, 0)
    
    return np.array(drag_dist), total_drag

# ==============================================================================
# GEOMETRY GENERATORS (Simplified for Visualization)
# ==============================================================================

def generate_nacelle_geometry(D_fan=0.125):
    """Generates the x,r coordinates for Spinner, Tailcone, and OML."""
    n_pts = 100
    
    # 1. Spinner (Elliptical)
    D_hub = D_fan * 0.3
    L_spin = D_hub * 1.0
    x_spin = np.linspace(-L_spin, 0, n_pts)
    r_spin = (D_hub/2) * np.sqrt(1 - (x_spin/L_spin)**2) 
    
    # 2. Tailcone (Parabolic)
    # Starts at L_fan_section. Let's say fan section is 3.53cm long.
    L_fan = 0.0353
    L_tail = D_hub * 2.5
    x_tail = np.linspace(L_fan, L_fan + L_tail, n_pts)
    # Parabola closing from R_hub to 0
    xi = (x_tail - L_fan) / L_tail
    r_tail = (D_hub/2) * (1 - xi**2)
    
    # 3. OML (Outer Mold Line)
    # Starts at Inlet Lip (-L_inlet) to Exit (+L_nozzle)
    L_inlet = D_fan * 0.6 * 1.3**0.5 
    L_nozz = D_fan * 1.0
    x_oml = np.linspace(-L_inlet, L_fan + L_nozz, n_pts)
    
    # Simplified NACA-like shape
    R_max = (D_fan/2) * 1.15
    R_inlet = (D_fan/2) * 1.02
    R_exit = (D_fan/2) * 0.9 
    
    r_oml = []
    L_total = x_oml[-1] - x_oml[0]
    for x in x_oml:
        t = (x - x_oml[0]) / L_total
        # Shape: Cubic hump
        if t < 0.4:
            tt = t / 0.4
            r = R_inlet + (R_max - R_inlet) * math.sin(tt * math.pi / 2)
        else:
            tt = (t - 0.4) / 0.6
            r = R_max - (R_max - R_exit) * math.sin(tt * math.pi / 2)
        r_oml.append(r)
        
    return (x_spin, r_spin), (x_tail, r_tail), (x_oml, r_oml)

# ==============================================================================
# MAIN COMPARISON LOGIC
# ==============================================================================

def run_comparison():
    print("Running Aerodynamic Drag Comparison: Takeoff vs Cruise...")
    
    # -- Conditions --
    # Case 1: Takeoff (Sea Level)
    V1 = 20.0 # m/s
    T1 = 25.0 # N
    rho1, mu1 = get_atmosphere(0)
    
    # Case 2: Cruise (10,000 ft ~ 3048 m)
    # Matches 'Constant Shaft Power' design point from previous analysis
    V2 = 80.0 # m/s
    T2 = 12.04 # N (Thrust required/available at P_shaft = const)
    rho2, mu2 = get_atmosphere(3048) 
    
    D_fan = 0.125
    
    # -- Geometry --
    (x_s, r_s), (x_t, r_t), (x_o, r_o) = generate_nacelle_geometry(D_fan)
    
    # -- Calculations Case 1 (Takeoff) --
    V_jet1 = calculate_jet_velocity(T1, V1, D_fan, rho1)
    
    # OML & Spinner see V_inf
    _, D_oml1 = calculate_component_drag(x_o, r_o, V1, rho1, mu1)
    _, D_spin1 = calculate_component_drag(x_s, r_s, V1, rho1, mu1)
    # Tailcone sees V_jet
    _, D_tail1 = calculate_component_drag(x_t, r_t, V_jet1, rho1, mu1)
    
    # -- Calculations Case 2 (Cruise) --
    V_jet2 = calculate_jet_velocity(T2, V2, D_fan, rho2)
    
    _, D_oml2 = calculate_component_drag(x_o, r_o, V2, rho2, mu2)
    _, D_spin2 = calculate_component_drag(x_s, r_s, V2, rho2, mu2)
    _, D_tail2 = calculate_component_drag(x_t, r_t, V_jet2, rho2, mu2)
    
    # Calculate Totals
    total_drag1 = D_oml1 + D_spin1 + D_tail1
    total_drag2 = D_oml2 + D_spin2 + D_tail2
    
    print(f"\n--- DRAG RESULTS ---")
    print(f"Takeoff (V={V1} m/s, Alt=0m): Total Drag = {total_drag1:.4f} N")
    print(f"  - OML: {D_oml1:.4f} N")
    print(f"  - Spinner: {D_spin1:.4f} N")
    print(f"  - Tailcone: {D_tail1:.4f} N (Jet V={V_jet1:.1f} m/s)")
    
    print(f"\nCruise (V={V2} m/s, Alt=3048m): Total Drag = {total_drag2:.4f} N")
    print(f"  - OML: {D_oml2:.4f} N")
    print(f"  - Spinner: {D_spin2:.4f} N")
    print(f"  - Tailcone: {D_tail2:.4f} N (Jet V={V_jet2:.1f} m/s)")
    
    # -- Plotting --
    fig = plt.figure(figsize=(14, 8))
    
    # 1. Visualization of Geometry
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(x_o, r_o, 'k-', linewidth=2, label='Nacelle OML')
    ax1.plot(x_o, [-r for r in r_o], 'k-', linewidth=2)
    
    ax1.fill_between(x_s, r_s, [-r for r in r_s], color='#3498db', alpha=0.5, label='Spinner (Inlet Flow)')
    ax1.fill_between(x_t, r_t, [-r for r in r_t], color='#e74c3c', alpha=0.5, label='Tailcone (Exhaust Flow)')
    
    ax1.set_aspect('equal')
    ax1.set_title("Component Identification")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Axial Position [m]")
    
    # 2. Drag Comparison Bar Chart
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    
    labels = ['Spinner', 'Tailcone', 'Nacelle OML']
    drag1 = [D_spin1, D_tail1, D_oml1]
    drag2 = [D_spin2, D_tail2, D_oml2]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, drag1, width, label=f'Takeoff ({V1} m/s)', color='#2ecc71')
    rects2 = ax2.bar(x + width/2, drag2, width, label=f'Cruise ({V2} m/s)', color='#e67e22')
    
    ax2.set_ylabel('Drag Force [N]')
    ax2.set_title(f'Drag Forces (Total Takeoff: {total_drag1:.2f}N | Cruise: {total_drag2:.2f}N)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, axis='y')
    
    # 3. Flow Condition Table
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.axis('off')
    
    table_data = [
        ["Parameter", f"Takeoff", f"Cruise"],
        ["Flight Speed", f"{V1:.1f} m/s", f"{V2:.1f} m/s"],
        ["Altitude", "0 m", "3048 m"],
        ["Jet Velocity", f"{V_jet1:.1f} m/s", f"{V_jet2:.1f} m/s"],
        ["Total Drag", f"{total_drag1:.3f} N", f"{total_drag2:.3f} N"],
        ["OML Drag %", f"{(D_oml1/total_drag1)*100:.1f}%", f"{(D_oml2/total_drag2)*100:.1f}%"]
    ]
    
    tbl = ax3.table(cellText=table_data, loc='center', cellLoc='center')
    tbl.scale(1, 2)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    ax3.set_title("Flow Conditions & Results")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()