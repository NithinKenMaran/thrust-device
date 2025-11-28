import adsk.core, adsk.fusion, adsk.cam, traceback
import math

# ==============================================================================
# MATH & GEOMETRY HELPERS (Standard Python - No Numpy/Matplotlib)
# ==============================================================================

def cubic_bezier(t, P0, P1, P2, P3):
    """Cubic Bezier curve for a given scalar t."""
    return ((1 - t)**3) * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + (t**3) * P3

def linspace(start, end, n):
    """Generates a list of n values from start to end."""
    if n < 2: return [start]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]

def rho_from_alt_m(alt_m):
    h_km = alt_m / 1000.0
    return 0.0037875*h_km**2 - 0.116575*h_km + 1.225

def design_power_from_T_V(T_des, rho, A, V_inf):
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

# ==============================================================================
# PROFILE GENERATORS (Adapted for List Outputs)
# ==============================================================================

def generate_bellmouth_profile(D_duct, D_inlet, L, n_points=60):
    R_throat = 0.5 * D_duct
    R_mouth  = 0.5 * D_inlet
    
    x_list = linspace(0.0, L, n_points)
    r_list = []
    
    for x in x_list:
        if L < 1e-9:
            r_list.append(R_throat)
        else:
            t = x / L
            f = 3.0 * t**2 - 2.0 * t**3
            r = R_throat + (R_mouth - R_throat) * f
            r_list.append(r)
            
    return x_list, r_list

def generate_spinner_profile(D_hub, L_over_D=1.0, n_points=60):
    R_hub = 0.5 * D_hub
    L = L_over_D * D_hub
    
    x_list = linspace(-L, 0, n_points)
    r_list = []
    
    for x in x_list:
        # Ellipse: (x/L)^2 + (r/R)^2 = 1 => r = R * sqrt(1 - (x/L)^2)
        val = 1 - (x/L)**2
        if val < 0: val = 0
        r = R_hub * math.sqrt(val)
        r_list.append(r)
        
    return x_list, r_list

def generate_tailcone_profile(D_hub, L_over_D=2.5, n_points=60):
    R_hub = 0.5 * D_hub
    L = L_over_D * D_hub
    
    x_list = linspace(0, L, n_points)
    r_list = []
    
    for x in x_list:
        # Parabolic closing
        r = R_hub * (1 - (x/L)**2)
        if r < 0: r = 0
        r_list.append(r)
        
    return x_list, r_list

def generate_nozzle_iml(D_duct, D_exit, L_nozzle, n_points=40):
    R_duct = D_duct / 2.0
    R_exit = D_exit / 2.0
    x_list = linspace(0, L_nozzle, n_points)
    r_list = []
    
    for x in x_list:
        t = x / L_nozzle
        # Bezier interpolation: P0=R_duct, P1=R_duct, P2=R_exit, P3=R_exit
        r = ((1 - t)**3) * R_duct + 3 * (1 - t)**2 * t * R_duct + 3 * (1 - t) * t**2 * R_exit + (t**3) * R_exit
        r_list.append(r)
        
    return x_list, r_list

def generate_nacelle_oml(D_max, L_total, D_exit, D_inlet, x_max_loc=0.35, n_points=100):
    x_list = linspace(0, L_total, n_points)
    r_list = []
    
    R_max = D_max / 2.0
    R_inlet = D_inlet / 2.0
    R_exit = D_exit / 2.0
    
    L_fore = L_total * x_max_loc
    L_aft = L_total - L_fore
    
    for x in x_list:
        r_val = 0.0
        if x <= L_fore:
            t = x / L_fore
            # Standard NACA forebody approx
            term = (1 - (1-t)**3)
            # pow(term, 1/3)
            if term < 0: term = 0
            shape = math.pow(term, 1.0/3.0)
            r_val = R_inlet + (R_max - R_inlet) * shape
            if r_val > R_max: r_val = R_max
        else:
            t = (x - L_fore) / L_aft
            r_val = cubic_bezier(t, R_max, R_max, (R_max+R_exit)/2.0, R_exit)
        
        r_list.append(r_val)
        
    return x_list, r_list

# ==============================================================================
# MAIN DESIGN LOGIC (Pure Python Port)
# ==============================================================================

def calculate_design_geometry():
    # --- Inputs ---
    D_duct = 0.125
    T_des = 25.0
    V_inf = 20.0
    alt_m = 0.0
    hub_ratio = 0.3
    n_points = 60
    
    # 1. Ambient & Fan Performance
    rho = rho_from_alt_m(alt_m)
    A_duct = math.pi * (D_duct**2) / 4.0
    P0, v_i, V2, V4, m_dot = design_power_from_T_V(T_des, rho, A_duct, V_inf)

    # 2. Inlet Optimization Loop (Simplified Grid Search)
    best_penalty = 1e9
    best_params = (1.3, 0.6) 

    cr_sweep = linspace(1.15, 1.5, 8)
    ld_sweep = linspace(0.4, 0.8, 8)

    for cr_val in cr_sweep:
        for ld_val in ld_sweep:
            D_inlet_local = math.sqrt(cr_val) * D_duct
            L_local = ld_val * D_inlet_local
            
            # Approx Wetted Area
            r1 = D_duct/2
            r2 = D_inlet_local/2
            slant_h = math.sqrt(L_local**2 + (r2 - r1)**2)
            surface_area = math.pi * (r1 + r2) * slant_h
            
            # Drag 1: Friction
            drag_friction = 0.5 * rho * (V_inf**2) * surface_area * 0.003
            
            # Drag 2: Lip Separation
            loss_factor = 0.01 / (ld_val**2 + 0.001) + 0.005 / ((cr_val-1.0)**2 + 0.01)
            q_duct = 0.5 * rho * V2**2
            drag_separation = q_duct * A_duct * loss_factor
            
            total = drag_friction + drag_separation
            if total < best_penalty:
                best_penalty = total
                best_params = (cr_val, ld_val)

    CR_inlet, L_inlet_over_D = best_params
    D_inlet = math.sqrt(CR_inlet) * D_duct
    L_inlet = L_inlet_over_D * D_inlet
    
    # 3. Nozzle Sizing
    if V4 > 0:
        A_exit = m_dot / (rho * V4)
    else:
        A_exit = A_duct 
    D_exit_nozzle = math.sqrt(4.0 * A_exit / math.pi)
    
    # 4. Lengths
    L_fan_section = 0.0353  # User Fixed
    L_nozzle = 1.0 * D_duct 
    D_hub = D_duct * hub_ratio
    
    # 5. Generate Profiles
    # A. Spinner
    x_spin, r_spin = generate_spinner_profile(D_hub, L_over_D=1.0, n_points=n_points)
    
    # B. Bellmouth (Inner) -> Flip and Offset
    x_bell_gen, r_bell_gen = generate_bellmouth_profile(D_duct, D_inlet, L_inlet, n_points)
    x_inlet_inner = [-val for val in reversed(x_bell_gen)]
    r_inlet_inner = list(reversed(r_bell_gen))
    
    # C. Fan Shroud
    x_fan = linspace(0, L_fan_section, int(n_points/2))
    r_fan = [D_duct/2.0] * len(x_fan)
    
    # D. Nozzle
    x_nozz_gen, r_nozz_gen = generate_nozzle_iml(D_duct, D_exit_nozzle, L_nozzle, n_points)
    x_nozzle = [val + L_fan_section for val in x_nozz_gen]
    r_nozzle = r_nozz_gen
    
    # E. Tail Cone
    x_tail_gen, r_tail_gen = generate_tailcone_profile(D_hub, L_over_D=2.5, n_points=n_points)
    x_tail = [val + L_fan_section for val in x_tail_gen]
    r_tail = r_tail_gen
    
    # F. OML
    Total_Axial_Length = L_inlet + L_fan_section + L_nozzle
    D_highlight = D_inlet * 1.02
    D_min_physical = D_duct + 2.0 * 0.004
    D_max_nacelle = max(D_min_physical, D_highlight * 1.05)
    D_exit_oml = D_exit_nozzle + 2.0 * 0.002
    
    x_oml_gen, r_oml_gen = generate_nacelle_oml(D_max_nacelle, Total_Axial_Length, D_exit_oml, D_highlight, 0.35, 100)
    x_oml = [val - L_inlet for val in x_oml_gen]
    r_oml = r_oml_gen
    
    # Combine Inner Walls
    x_inner = x_inlet_inner + x_fan + x_nozzle
    r_inner = r_inlet_inner + r_fan + r_nozzle
    
    return {
        "x_oml": x_oml, "r_oml": r_oml,
        "x_inner": x_inner, "r_inner": r_inner,
        "x_spin": x_spin, "r_spin": r_spin,
        "x_tail": x_tail, "r_tail": r_tail,
        "x_hub": x_fan, "r_hub": [r_spin[-1]] * len(x_fan) # Hub connection
    }

# ==============================================================================
# FUSION 360 DRAWING LOGIC
# ==============================================================================

def draw_spline(sketch, x_vals, r_vals):
    points = adsk.core.ObjectCollection.create()
    for x, r in zip(x_vals, r_vals):
        # Fusion units are cm, Simulation is m. Multiply by 100.
        # Sketch Plane: x_sim -> x_fusion, r_sim -> y_fusion
        points.add(adsk.core.Point3D.create(x * 100.0, r * 100.0, 0))
    
    sketch.sketchCurves.sketchFittedSplines.add(points)

def draw_line(sketch, x1, r1, x2, r2, is_construction=False):
    p1 = adsk.core.Point3D.create(x1 * 100.0, r1 * 100.0, 0)
    p2 = adsk.core.Point3D.create(x2 * 100.0, r2 * 100.0, 0)
    lines = sketch.sketchCurves.sketchLines
    line = lines.addByTwoPoints(p1, p2)
    if is_construction:
        line.isConstruction = True

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        design = app.activeProduct

        # Get the root component of the active design
        rootComp = design.rootComponent

        # Create a new sketch on the XY plane
        sketches = rootComp.sketches
        xyPlane = rootComp.xYConstructionPlane
        sketch = sketches.add(xyPlane)
        sketch.name = "EDF Nacelle Profile"

        # --- Calculate Geometry ---
        data = calculate_design_geometry()
        
        # --- Draw Curves ---
        
        # 1. Outer Mold Line (OML)
        draw_spline(sketch, data["x_oml"], data["r_oml"])
        
        # 2. Inner Duct Wall (Inlet -> Fan -> Nozzle)
        draw_spline(sketch, data["x_inner"], data["r_inner"])
        
        # 3. Centerbody (Spinner)
        draw_spline(sketch, data["x_spin"], data["r_spin"])
        
        # 4. Centerbody (Hub - Straight Line)
        draw_line(sketch, data["x_hub"][0], data["r_hub"][0], data["x_hub"][-1], data["r_hub"][-1])

        # 5. Centerbody (Tail Cone)
        draw_spline(sketch, data["x_tail"], data["r_tail"])

        # 6. Centerline (Construction Line for Revolve)
        min_x = data["x_oml"][0]
        max_x = data["x_oml"][-1]
        draw_line(sketch, min_x, 0, max_x, 0, is_construction=True)
        
        ui.messageBox('EDF Nacelle Sketch Generated Successfully!\nUse the "Revolve" tool to create the solid body.')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
