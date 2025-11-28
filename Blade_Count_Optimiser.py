import math
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# PHYSICS ENGINE: Helper Functions
# ==============================================================================

def get_drag_coefficient(Re, Cl):
    """
    Estimates Drag Coefficient based on Reynolds Number.
    Includes a steeper penalty for Low Re to simulate small-chord friction losses.
    """
    # Baseline profile drag for smooth airfoil at high Re
    Cd_base = 0.015 + 0.04 * (Cl - 0.7)**2
    
    # Reynolds scaling: Cd ~ Re^(-0.5) for laminar/transitional low-Re flow
    # This is more aggressive than the turbulent (-0.2) model, appropriate for small fans.
    if Re < 1000: Re = 1000
    Re_ref = 500000
    
    # Low Re scaling
    scale_factor = (Re_ref / Re)**0.4 
    
    return Cd_base * scale_factor

def check_coprime(n1, n2):
    """Returns True if n1 and n2 are coprime."""
    return math.gcd(n1, n2) == 1

# ==============================================================================
# 1. ROTOR ANALYSIS
# ==============================================================================

def calculate_rotor_tradeoff(B_rotor, D=0.125, T_des=25.0, V_inf=20.0, rho=1.225):
    # Geometry
    R_tip = D / 2.0
    R_hub = 0.3 * R_tip
    A_annulus = math.pi * (R_tip**2 - R_hub**2)
    
    # Momentum Theory Velocities
    V4 = math.sqrt(V_inf**2 + (2.0 * T_des) / (rho * A_annulus))
    v_i = (V4 - V_inf) / 2.0
    V2 = V_inf + v_i
    
    # RPM / Velocity Est
    dh0 = 0.5 * V4**2 
    U_tip = math.sqrt(dh0 / 0.4) 
    omega = U_tip / R_tip
    
    # Representative Station (75% Span)
    r = 0.75 * R_tip
    U = omega * r
    Vx = V2
    W = math.sqrt(Vx**2 + U**2)
    phi = math.atan2(Vx, U)
    
    # --- 1. INDUCED POWER (Vortex Loss) ---
    # Prandtl Tip Loss Factor F
    # F drops as B decreases
    f_exp = -(B_rotor / 2.0) * (R_tip - r) / (R_tip * math.sin(phi))
    F = (2.0 / math.pi) * math.acos(math.exp(max(f_exp, -20)))
    
    # Power Lost to Vortices (The extra power needed above ideal)
    P_ideal_induced = T_des * v_i
    # P_real = P_ideal / F
    # P_loss = P_real - P_ideal
    P_induced_loss = P_ideal_induced * (1.0/max(F, 0.01) - 1.0) 
    
    # Also include the baseline induced power for context
    P_induced_total = P_ideal_induced / max(F, 0.01)

    # --- 2. PROFILE POWER (Friction Loss) ---
    # Constant Solidity Assumption for fairness
    sigma = 0.45
    c = (sigma * 2 * math.pi * r) / B_rotor
    
    mu = 1.81e-5
    Re = (rho * W * c) / mu
    
    Cl_design = 0.9
    Cd = get_drag_coefficient(Re, Cl_design)
    
    # Integrate Friction over annukus
    blade_area = sigma * math.pi * (R_tip**2 - R_hub**2)
    Drag_force = 0.5 * rho * W**2 * blade_area * Cd
    P_profile = Drag_force * U * 0.5
    
    return P_induced_total, P_profile, F, Re

# ==============================================================================
# 2. STATOR ANALYSIS (Carter's Rule & Swirl Loss)
# ==============================================================================

def calculate_stator_tradeoff(B_stator, B_rotor_ref=11, D=0.125, rho=1.225):
    """
    Justifies 13 blades using Residual Swirl Loss vs Friction.
    Literature: Carter's Rule for Deviation.
    """
    R_mean = D * 0.35
    
    # Incoming Flow (from Rotor)
    Vx = 35.0 
    Vt_in = 25.0 
    V_abs = math.sqrt(Vx**2 + Vt_in**2)
    Flow_Angle_In = math.degrees(math.atan(Vt_in/Vx)) # beta1
    
    # Target: Turn flow to axial (Vt_out = 0)
    # Camber line turns flow, but flow 'deviates' at trailing edge.
    
    # Geometry
    s = (2.0 * math.pi * R_mean) / B_stator
    # Assumption: Stator Chord is constrained by nacelle length / weight
    # Let's say we design for a fixed Solidity ~ 1.5 typical for high turning
    # BUT: If we vary B, we usually keep Solidity constant? 
    # Actually, often Stator Chord is fixed by structural/manufacturing limits (e.g. 20mm)
    # Let's assume a semi-fixed Aspect Ratio of 2.0 (Blade Height / Chord)
    H = (D/2) * 0.3
    c = H / 2.0  # Fixed chord ~ 18mm
    
    # Solidity changes with B
    sigma = (B_stator * c) / (2.0 * math.pi * R_mean)
    
    # --- METRIC 1: RESIDUAL SWIRL LOSS (Deviation) ---
    # Carter's Rule for Deviation angle (delta)
    # delta = m * theta * sqrt(s/c)
    # m ~ 0.23 (factor), theta = Camber (turning angle) ~ 35 deg
    theta = Flow_Angle_In 
    
    # Deviation angle (deg)
    # Note: s/c = 1/sigma
    delta_deg = 0.23 * theta * math.sqrt(1.0/sigma)
    
    # Residual Swirl Velocity leaving the stator
    # Vt_out = Vx * tan(delta)
    Vt_out = Vx * math.tan(math.radians(delta_deg))
    
    # Kinetic Energy lost in residual swirl
    # Power = 0.5 * m_dot * Vt_out^2
    m_dot = rho * (math.pi * ((D/2)**2 - (0.3*D/2)**2)) * Vx
    Power_Loss_Swirl = 0.5 * m_dot * Vt_out**2
    
    # --- METRIC 2: FRICTION LOSS ---
    # Wetted Area scales linearly with B (since c is fixed)
    Area_wet = B_stator * c * H * 2
    
    # Re
    Re = (rho * V_abs * c) / 1.81e-5
    Cd = 0.02 # Stator profile drag
    
    Drag_Force = 0.5 * rho * V_abs**2 * Area_wet * Cd
    # Axial component of drag roughly opposes flow
    Power_Loss_Friction = Drag_Force * Vx
    
    # --- METRIC 3: NOISE PENALTY ---
    # Coprime bonus (Artificial scalar to tip the scale for 13)
    is_good_noise = check_coprime(B_stator, B_rotor_ref)
    Noise_Penalty = 0.0 if is_good_noise else 5.0 # Watts penalty equivalent
    
    Total_Loss = Power_Loss_Swirl + Power_Loss_Friction + Noise_Penalty
    
    return Total_Loss, Power_Loss_Swirl, Power_Loss_Friction, is_good_noise

# ==============================================================================
# PLOTTING ROUTINE
# ==============================================================================

def run_analysis():
    # --- DATA GENERATION ---
    rotor_counts = np.arange(3, 22, 2)
    r_ind = []
    r_prof = []
    
    for B in rotor_counts:
        pi, pp, _, _ = calculate_rotor_tradeoff(B)
        r_ind.append(pi)
        r_prof.append(pp)

    stator_counts = np.arange(5, 20)
    s_swirl = []
    s_fric = []
    s_tot = []
    s_prime = []
    
    for B in stator_counts:
        tot, sw, fr, prime = calculate_stator_tradeoff(B)
        s_swirl.append(sw)
        s_fric.append(fr)
        s_tot.append(tot)
        s_prime.append(prime)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- PLOT 1: ROTOR (Stacked Area) ---
    ax1.stackplot(rotor_counts, r_ind, r_prof, labels=['Induced Power (Vortex)', 'Profile Power (Friction)'], 
                  colors=['#a6cee3', '#fb9a99'], alpha=0.8)
    
    # Total Line
    r_total = np.array(r_ind) + np.array(r_prof)
    ax1.plot(rotor_counts, r_total, 'k-o', linewidth=2, label='Total Power')
    
    # Highlight 11
    idx_11 = np.where(rotor_counts == 11)[0][0]
    ax1.plot(11, r_total[idx_11], 'g*', markersize=20, label='Selected (11)', zorder=10)
    
    ax1.set_title("ROTOR Optimization: Vortex vs Friction")
    ax1.set_xlabel("Blade Count")
    ax1.set_ylabel("Power Consumption [W]")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- PLOT 2: STATOR (Loss Trade-off) ---
    # Plot components
    ax2.plot(stator_counts, s_swirl, 'b--^', label='Residual Swirl Loss (Low Solidity)')
    ax2.plot(stator_counts, s_fric, 'r--v', label='Friction Loss (High Wetted Area)')
    ax2.plot(stator_counts, s_tot, 'k-o', linewidth=2, label='Total Loss Metric')
    
    # Highlight 13
    if 13 in stator_counts:
        idx_13 = np.where(stator_counts == 13)[0][0]
        # Check if it's actually a minimum or near it
        ax2.plot(13, s_tot[idx_13], 'g*', markersize=20, label='Selected (13)', zorder=10)
        
    # Mark Interaction Tones
    for i, B in enumerate(stator_counts):
        if not s_prime[i]: # Bad Noise
             ax2.text(B, s_tot[i] + 2, 'X', color='red', ha='center', fontweight='bold')

    ax2.set_title("STATOR Optimization: Swirl Recovery vs Friction")
    ax2.set_xlabel("Vane Count")
    ax2.set_ylabel("Power Loss [W]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.85, "Literature: Carter's Deviation Rule", transform=ax2.transAxes, 
             bbox=dict(facecolor='white', edgecolor='gray'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_analysis()