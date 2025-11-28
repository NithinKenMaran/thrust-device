import math
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# STRUCTURAL ANALYSIS ENGINE
# ==============================================================================

def calculate_moment_of_inertia(chord, thickness):
    """
    Calculates Area Moment of Inertia for the stator blade cross-section.
    Approximated as a simplified Airfoil/Rectangle for conservative stress analysis.
    
    I_x (Flapwise) resists Thrust bending.
    I_y (Edgewise) resists Torque bending.
    """
    # Rectangle approximation (w=chord, h=thickness)
    # I = (b * h^3) / 12
    
    # Flapwise (Bending up/down due to Thrust) -> Thickness is the 'h' dimension
    I_flap = (chord * thickness**3) / 12.0
    
    # Edgewise (Bending side-to-side due to Torque) -> Chord is the 'h' dimension
    I_edge = (thickness * chord**3) / 12.0
    
    Area = chord * thickness
    return I_flap, I_edge, Area

def analyze_stator_stress(
    Thrust_N=25.0, 
    Torque_Nm=0.9, 
    N_stators=13, 
    R_hub=0.018, 
    R_tip=0.0625, 
    Chord=0.018, 
    Thickness_max=0.003
):
    """
    Performs stress analysis on Stator Blades acting as cantilever beams holding the motor.
    """
    print(f"--- Stator Blade Structural Analysis ({N_stators} Blades) ---")
    
    # 1. GEOMETRY
    L_beam = R_tip - R_hub
    
    # Cross-section properties
    t_avg = Thickness_max * 0.8 
    c_avg = Chord
    I_flap, I_edge, Area = calculate_moment_of_inertia(c_avg, t_avg)
    
    # 2. LOAD DISTRIBUTION
    F_thrust_per_blade = Thrust_N / N_stators
    F_tangential_total = Torque_Nm / R_hub
    F_tangential_per_blade = F_tangential_total / N_stators
    
    # 3. STRESS CALCULATION (Cantilever Beam)
    # Max Moment occurs at the Root
    
    M_flap = F_thrust_per_blade * L_beam
    M_edge = F_tangential_per_blade * L_beam
    
    y_flap = t_avg / 2.0
    y_edge = c_avg / 2.0
    
    sigma_bending_thrust = (M_flap * y_flap) / I_flap
    sigma_bending_torque = (M_edge * y_edge) / I_edge
    
    # Combined Stress (Von Mises approx for max fiber stress)
    sigma_normal_max = sigma_bending_thrust + sigma_bending_torque
    
    # 4. FACTOR OF SAFETY
    # Material: ABS (Acrylonitrile Butadiene Styrene)
    Yield_Strength_Pa = 40e6 
    
    FOS = Yield_Strength_Pa / sigma_normal_max
    
    print(f"  Max Stress Estimates (Root):")
    print(f"    Bending (Thrust): {sigma_bending_thrust/1e6:.2f} MPa")
    print(f"    Bending (Torque): {sigma_bending_torque/1e6:.2f} MPa")
    print(f"    Total Max Stress: {sigma_normal_max/1e6:.2f} MPa")
    print(f"  Material: ABS (Yield: 40 MPa)")
    print(f"  Stator FOS: {FOS:.2f}")
    
    return {
        'sigma_thrust': sigma_bending_thrust,
        'sigma_torque': sigma_bending_torque,
        'sigma_total': sigma_normal_max,
        'yield': Yield_Strength_Pa,
        'fos': FOS
    }

def analyze_bolt_connection(
    Drag_Cone_N=0.5, 
    Bolt_Size="M2", 
    N_bolts=4, 
    Engagement_Len_mm=5,
    Insert_OD_mm=3.5
):
    """
    FOS Analysis for the bolts holding the exhaust cone.
    Configuration: Steel Bolt M2 + Brass Heat-Set Insert + ABS Hub
    """
    print(f"\n--- Exhaust Cone Bolt Analysis ({N_bolts}x {Bolt_Size} with Brass Inserts) ---")
    
    # 1. BOLT GEOMETRY
    d_major = 2.0 # mm
    pitch = 0.4 # mm
    A_stress = (math.pi/4) * (d_major - 0.938*pitch)**2 # mm^2
    A_stress_m2 = A_stress * 1e-6
    
    # 2. LOADS
    Force_per_bolt = Drag_Cone_N / N_bolts
    
    # 3. FAILURE MODE 1: BOLT TENSION
    Yield_Bolt = 640e6 # Steel 8.8
    sigma_bolt = Force_per_bolt / A_stress_m2
    FOS_Bolt = Yield_Bolt / sigma_bolt
    
    # 4. FAILURE MODE 2: BRASS THREAD STRIPPING
    Shear_Strength_Brass = 200e6 
    L_engage = Engagement_Len_mm / 1000.0
    d_major_m = d_major / 1000.0
    A_shear_thread = math.pi * d_major_m * 0.75 * L_engage 
    
    tau_thread = Force_per_bolt / A_shear_thread
    FOS_Thread = Shear_Strength_Brass / tau_thread
    
    # 5. FAILURE MODE 3: INSERT PULL-OUT (ABS SHEAR) [CRITICAL]
    Shear_Strength_ABS = 25e6 # 25 MPa
    
    Insert_OD_m = Insert_OD_mm / 1000.0
    A_shear_pullout = math.pi * Insert_OD_m * L_engage
    
    tau_pullout = Force_per_bolt / A_shear_pullout
    FOS_Pullout = Shear_Strength_ABS / tau_pullout
    
    print(f"  Load per Bolt: {Force_per_bolt:.3f} N")
    print(f"  [1] Bolt Tensile Stress: {sigma_bolt/1e6:.4f} MPa (FOS: {FOS_Bolt:.1f})")
    print(f"  [2] Brass Thread Shear : {tau_thread/1e6:.4f} MPa (FOS: {FOS_Thread:.1f})")
    print(f"  [3] ABS Pull-Out Stress: {tau_pullout/1e6:.4f} MPa (FOS: {FOS_Pullout:.1f})")
    
    return {
        'fos_bolt': FOS_Bolt,
        'fos_thread': FOS_Thread,
        'fos_pullout': FOS_Pullout,
        'stress_pullout': tau_pullout
    }

# ==============================================================================
# VISUALIZATION DASHBOARD
# ==============================================================================

def visualize_structural_report(stator_res, bolt_res):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Structural Integrity Analysis: Stator & Exhaust Cone', fontsize=16)
    
    # --- SUBPLOT 1: STATOR STRESSES ---
    ax1 = fig.add_subplot(1, 2, 1)
    
    components = ['Thrust Bending', 'Torque Bending']
    stress_vals = [stator_res['sigma_thrust']/1e6, stator_res['sigma_torque']/1e6]
    total_stress = stator_res['sigma_total']/1e6
    yield_stress = stator_res['yield']/1e6
    
    # Plot Components
    bars = ax1.bar(components, stress_vals, color=['#3498db', '#9b59b6'], alpha=0.8)
    
    # Plot Total as a separate bar or line? Let's use a bar next to them
    ax1.bar(['Total Max Stress'], [total_stress], color='#e74c3c', alpha=0.9)
    
    # Yield Line
    ax1.axhline(yield_stress, color='k', linestyle='--', linewidth=2, label=f'ABS Yield ({yield_stress} MPa)')
    
    # Annotations
    ax1.set_ylabel('Stress [MPa]')
    ax1.set_title(f'Stator Blade Root Stress (FOS: {stator_res["fos"]:.2f})')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Value labels
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    # --- SUBPLOT 2: BOLT SAFETY FACTORS ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    modes = ['Bolt Tension', 'Thread Shear', 'Insert Pull-out']
    fos_values = [bolt_res['fos_bolt'], bolt_res['fos_thread'], bolt_res['fos_pullout']]
    
    # Color Logic
    colors = []
    for f in fos_values:
        if f < 2.0: colors.append('#e74c3c') # Red (Danger)
        elif f < 4.0: colors.append('#f1c40f') # Yellow (Caution)
        else: colors.append('#2ecc71') # Green (Safe)
        
    y_pos = np.arange(len(modes))
    bars2 = ax2.barh(y_pos, fos_values, color=colors, edgecolor='black')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(modes)
    ax2.set_xlabel('Factor of Safety (Log Scale)')
    ax2.set_title('Connection Failure Modes')
    ax2.set_xscale('log') # Log scale because FOS varies wildly (Steel vs Plastic)
    
    # Limit Line
    ax2.axvline(3.0, color='gray', linestyle=':', linewidth=2)
    ax2.text(3.2, -0.5, 'Min Recommended FOS (3.0)', color='gray')
    
    # Annotate
    for i, v in enumerate(fos_values):
        ax2.text(v * 1.1, i, f'{v:.1f}', va='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==============================================================================
# MAIN RUNNER
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Run Stator Analysis ---
    stator_results = analyze_stator_stress(
        Thrust_N=30.0, 
        Torque_Nm=1.0, 
        N_stators=13, 
        R_hub=0.0188, 
        R_tip=0.0625, 
        Chord=0.018,    
        Thickness_max=0.003 
    )
    
    # --- 2. Run Bolt Analysis ---
    bolt_results = analyze_bolt_connection(
        Drag_Cone_N=2.0, 
        Bolt_Size="M2", 
        N_bolts=4, 
        Engagement_Len_mm=5, # 5mm insert length
        Insert_OD_mm=3.5     
    )
    
    # --- 3. Visualize ---
    visualize_structural_report(stator_results, bolt_results)