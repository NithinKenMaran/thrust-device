import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# PHYSICS ENGINE
# ==============================================================================

def prandtl_tip_loss(r, R_tip, B, phi_rad):
    """
    Standard Prandtl Tip Loss Function for Open Rotors.
    F goes to 0 at the tip.
    """
    if r >= R_tip: return 0.0
    
    f_exp = -(B / 2.0) * (R_tip - r) / (R_tip * math.sin(phi_rad))
    
    # Clamp exponent
    f_exp = max(f_exp, -50)
    
    F = (2.0 / math.pi) * math.acos(math.exp(f_exp))
    return F

def ducted_tip_loss(r, R_tip, B, phi_rad, gap_mm):
    """
    Modified Tip Loss for Ducted Fan.
    The duct wall suppresses the vortex, making the blade act like it has 
    an 'Effective Radius' larger than the physical radius.
    """
    # Virtual extension of the blade due to wall presence
    R_gap = R_tip + (gap_mm / 1000.0)
    
    # Check if we are in the physical blade region
    if r > R_tip: return 0.0
    
    # Effective Tip Radius for Prandtl Calculation
    f_exp = -(B / 2.0) * (R_gap - r) / (R_gap * math.sin(phi_rad))
    
    # Duct suppression factor
    suppression_factor = 3.0 
    
    F = (2.0 / math.pi) * math.acos(math.exp(f_exp * suppression_factor))
    
    return F

def calculate_circulation_distribution(D=0.125, B=11, gap_mm=0.5):
    """
    Calculates the lift/circulation distribution for Open vs Ducted.
    """
    R_tip = D / 2.0
    R_hub = 0.3 * R_tip
    
    # Stations
    r_arr = np.linspace(R_hub, R_tip, 200)
    
    # Setup Flow Conditions (Simplified for Visualization)
    phi_deg = 30.0
    phi_rad = math.radians(phi_deg)
    
    F_open = []
    F_ducted = []
    
    Gamma_ideal = 1.0 
    
    Gamma_open = []
    Gamma_ducted = []
    
    for r in r_arr:
        # 1. Open Rotor
        F_o = prandtl_tip_loss(r, R_tip, B, phi_rad)
        F_open.append(F_o)
        Gamma_open.append(Gamma_ideal * F_o)
        
        # 2. Ducted Rotor
        F_d = ducted_tip_loss(r, R_tip, B, phi_rad, gap_mm)
        F_ducted.append(F_d)
        Gamma_ducted.append(Gamma_ideal * F_d)
        
    return r_arr, F_open, F_ducted, Gamma_open, Gamma_ducted

# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def visualize_duct_effect():
    print("Visualizing Ducted Fan vs Open Propeller Tip Vortex Effect...")
    
    # Inputs
    D = 0.125 # 125mm
    B = 11    # 11 Blades
    Gap = 0.5 # 0.5mm Tip Clearance
    
    r, F_o, F_d, G_o, G_d = calculate_circulation_distribution(D, B, Gap)
    
    # Normalize Radius
    r_norm = r / (D/2.0)
    
    # Calculate "Recovered" Lift Potential (Area under curve)
    area_open = np.trapz(G_o, r_norm)
    area_ducted = np.trapz(G_d, r_norm)
    improvement = ((area_ducted - area_open) / area_open) * 100
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: Tip Loss Factor F (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(r_norm, F_d, 'g-', linewidth=2.5, label=f'Ducted (Gap {Gap}mm)')
    ax1.plot(r_norm, F_o, 'r--', linewidth=2, label='Open Propeller')
    
    ax1.fill_between(r_norm, F_o, F_d, color='green', alpha=0.1, label='Recovered Potential')
    
    ax1.set_title(f"Tip Loss Factor F (B={B})")
    ax1.set_ylabel("Factor F (1.0 = Ideal)")
    ax1.set_xlabel("Normalized Radius (r/R)")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_xlim(0.3, 1.02)
    ax1.set_ylim(0, 1.1)

    # Plot 2: Load Distribution (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    
    # Focus on outer 20%
    zoom_idx = np.where(r_norm > 0.8)[0]
    
    ax2.plot(r_norm[zoom_idx], np.array(G_d)[zoom_idx], 'g-', linewidth=2.5, label='Ducted Load')
    ax2.plot(r_norm[zoom_idx], np.array(G_o)[zoom_idx], 'r--', linewidth=2, label='Open Load')
    
    ax2.set_title("Blade Loading Near Tip (Zoomed)")
    ax2.set_xlabel("Normalized Radius (r/R)")
    ax2.set_ylabel("Normalized Circulation")
    ax2.grid(True, alpha=0.5)
    ax2.legend(fontsize=9)
    
    # Text Stats Box
    stats_text = (
        f"PERFORMANCE GAIN:\n"
        f"Calculated Lift Increase: +{improvement:.1f}%\n"
        f"Due to Vortex Suppression"
    )
    ax2.text(0.05, 0.1, stats_text, transform=ax2.transAxes, 
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9), fontsize=9)
    
    # Plot 3: Blade Schematic (Bottom)
    ax3 = plt.subplot(2, 1, 2)
    
    # Dimensions for Schematic (mm)
    R_tip_mm = D * 1000 / 2
    R_hub_mm = R_tip_mm * 0.3
    Gap_mm = Gap
    Chord_Root = 25
    Chord_Tip = 15
    
    # Draw Hub
    ax3.fill_between([-10, 40], [0, 0], [R_hub_mm, R_hub_mm], color='#95a5a6', alpha=0.3, label='Hub')
    
    # Draw Blade (Trapezoid representation)
    blade_poly = [
        [0, R_hub_mm],                  # Root LE
        [Chord_Root, R_hub_mm],         # Root TE
        [Chord_Root - (Chord_Root-Chord_Tip)/2, R_tip_mm], # Tip TE
        [(Chord_Root-Chord_Tip)/2, R_tip_mm]               # Tip LE
    ]
    blade_patch = patches.Polygon(blade_poly, closed=True, facecolor='#2ecc71', edgecolor='#27ae60', alpha=0.7, label='Rotor Blade')
    ax3.add_patch(blade_patch)
    
    # Draw Duct Wall
    duct_r = R_tip_mm + Gap_mm
    ax3.fill_between([-10, 40], [duct_r, duct_r], [duct_r+10, duct_r+10], color='#34495e', alpha=0.8, label='Duct Wall')
    
    # Highlight Gap
    ax3.plot([(Chord_Root-Chord_Tip)/2, (Chord_Root-Chord_Tip)/2], [R_tip_mm, duct_r], 'r-', linewidth=1)
    ax3.text((Chord_Root-Chord_Tip)/2 - 5, R_tip_mm + Gap_mm + 2, f"Tip Gap: {Gap_mm} mm", color='red', fontsize=10, fontweight='bold')
    
    # Styling
    ax3.set_xlim(-5, 35)
    ax3.set_ylim(0, R_tip_mm + 15)
    ax3.set_aspect('equal')
    ax3.set_xlabel('Axial Position [mm]')
    ax3.set_ylabel('Radial Position [mm]')
    ax3.set_title('Schematic: Blade Span & Tip Gap Context')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_duct_effect()