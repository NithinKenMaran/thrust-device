import adsk.core, adsk.fusion, adsk.cam, traceback
import math

# ============================================================
# Utility math (no numpy)
# ============================================================

def alpha_from_CL(CL_target, slope_per_deg, alpha0_deg):
    return CL_target / slope_per_deg + alpha0_deg


def cubic_bezier(t, P0, P1, P2, P3):
    return ((1 - t)**3) * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + (t**3) * P3


def smooth_list_py(vals, w=0.3):
    out = vals[:]
    n = len(vals)
    for i in range(1, n - 1):
        out[i] = (1.0 - w) * vals[i] + 0.5 * w * (vals[i - 1] + vals[i + 1])
    return out


def cd_from_cl_single(CL, cl_opt=0.7, cd_min=0.015, k2=0.04):
    return cd_min + k2 * (CL - cl_opt)**2


# ============================================================
# NACA 4-digit airfoil generator (for 4412 / 0012)
# ============================================================

def generate_naca4(digits="4412", n_points=101):
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError("digits must be '4412', '0012', etc.")
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0

    x_list, yt_list, yc_list, dyc_dx_list = [], [], [], []

    for i in range(n_points):
        theta = float(i) * math.pi / float(n_points - 1)
        x = 0.5 * (1.0 - math.cos(theta))  # cosine spacing
        x_list.append(x)

    for x in x_list:
        yt = 5.0 * t * (
            0.2969 * math.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
        if p == 0:
            yc = 0.0
            dyc_dx = 0.0
        elif x < p:
            yc = m / (p ** 2) * (2.0 * p * x - x**2)
            dyc_dx = 2.0 * m / (p ** 2) * (p - x)
        else:
            yc = m / ((1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2)
            dyc_dx = 2.0 * m / ((1.0 - p) ** 2) * (p - x)

        yt_list.append(yt)
        yc_list.append(yc)
        dyc_dx_list.append(dyc_dx)

    xu, yu, xl, yl = [], [], [], []
    for x, yt, yc, dyc_dx in zip(x_list, yt_list, yc_list, dyc_dx_list):
        theta_c = math.atan(dyc_dx)
        xu.append(x - yt * math.sin(theta_c))
        yu.append(yc + yt * math.cos(theta_c))
        xl.append(x + yt * math.sin(theta_c))
        yl.append(yc - yt * math.cos(theta_c))

    x_coords = list(reversed(xu)) + xl[1:]
    y_coords = list(reversed(yu)) + yl[1:]
    return x_coords, y_coords


# ============================================================
# Rotor design (base + BEMT chord scaling)
# ============================================================

def design_rotor_scaled(
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
    R_tip = 0.5 * D
    R_hub = hub_to_tip * R_tip
    A_annulus = math.pi * (R_tip**2 - R_hub**2)
    r_m = math.sqrt(0.5 * (R_hub**2 + R_tip**2))

    V4 = math.sqrt(2.0 * T_des / (rho * A_annulus))
    dh0 = 0.5 * V4**2

    U_m = math.sqrt(dh0 / psi_des)
    omega = U_m / r_m
    rpm = omega * 60.0 / (2.0 * math.pi)

    Vx_const = phi_mean * U_m

    # radial grid
    r = []
    r_R = []
    r_start = R_hub * 1.03
    r_end = R_tip * 0.99
    for i in range(n_span):
        s = float(i) / float(n_span - 1) if n_span > 1 else 0.0
        ri = r_start + (r_end - r_start) * s
        r.append(ri)
        r_R.append(ri / R_tip)

    U = []
    Vx = []
    Vtheta1 = []
    Vtheta2 = []
    V1 = []
    V2 = []
    beta1 = []
    beta2 = []

    for ri in r:
        Ui = omega * ri
        U.append(Ui)
        Vxi = Vx_const
        Vx.append(Vxi)
        Vt1 = 0.0
        Vtheta1.append(Vt1)
        Vt2 = dh0 / Ui
        Vtheta2.append(Vt2)
        V1i = math.sqrt(Vxi**2 + Vt1**2)
        V2i = math.sqrt(Vxi**2 + Vt2**2)
        V1.append(V1i)
        V2.append(V2i)
        b1 = math.degrees(math.atan2(Vxi, Ui - Vt1))
        b2 = math.degrees(math.atan2(Vxi, Ui - Vt2))
        beta1.append(b1)
        beta2.append(b2)

    # CL target and AoA
    a_rotor = 0.11
    alpha0_rotor = -2.0
    alpha_section = alpha_from_CL(CL_target, a_rotor, alpha0_rotor)
    CL = [CL_target for _ in r]
    alpha = [alpha_section for _ in r]

    # Bezier chord
    c_pref = []
    cR_root = 0.18
    cR_tip  = 0.06
    cR1     = 0.16
    cR2     = 0.10

    for ri in r:
        t = (ri - R_hub) / (R_tip - R_hub)
        cR = cubic_bezier(t, cR_root, cR1, cR2, cR_tip)
        if cR < cR_min:
            cR = cR_min
        if cR > cR_max:
            cR = cR_max
        c_pref.append(cR * R_tip)

    # DF-based minimum chord
    c_from_DF = [0.0 for _ in r]
    for i in range(len(r)):
        ri = r[i]
        s = 2.0 * math.pi * ri / float(B_rotor)
        V1i = V1[i]
        V2i = V2[i]
        Vt1 = Vtheta1[i]
        Vt2 = Vtheta2[i]
        if V1i <= 1e-6:
            continue
        dVtheta = Vt2 - Vt1
        if abs(dVtheta) < 1e-8:
            continue
        s_over_c_req = (DF_limit - 1.0 + V2i / V1i) * (2.0 * V1i / dVtheta)
        if s_over_c_req > 0:
            c_from_DF[i] = s / s_over_c_req

    c = []
    for i in range(len(r)):
        ci = c_pref[i]
        if c_from_DF[i] > ci:
            ci = c_from_DF[i]
        if ci > cR_max * R_tip:
            ci = cR_max * R_tip
        c.append(ci)

    c = smooth_list_py(c, w=0.3)

    # BEMT thrust for base rotor
    T_bemt0, Q_bemt0, D_prof0 = bemt_for_rotor(
        r, U, Vx, beta1, c, CL, rho, B_rotor, CL_target
    )

    if T_bemt0 <= 1e-6:
        scale_c = 1.0
    else:
        scale_c = T_des / T_bemt0

    c_scaled = [ci * scale_c for ci in c]

    # DF for scaled chord
    DF_scaled = []
    for i in range(len(r)):
        ri = r[i]
        s = 2.0 * math.pi * ri / float(B_rotor)
        V1i = V1[i]
        V2i = V2[i]
        Vt1 = Vtheta1[i]
        Vt2 = Vtheta2[i]
        ci = c_scaled[i]
        if ci <= 1e-9 or V1i <= 1e-6:
            DF_scaled.append(0.0)
            continue
        s_over_c = s / ci
        DF_val = 1.0 - V2i / V1i + (Vt2 - Vt1) / (2.0 * V1i) * s_over_c
        DF_scaled.append(DF_val)

    # BEMT again for scaled rotor
    T_bemt, Q_bemt, D_prof = bemt_for_rotor(
        r, U, Vx, beta1, c_scaled, CL, rho, B_rotor, CL_target
    )

    return {
        "r": r,
        "r_R": r_R,
        "c": c_scaled,
        "beta1": beta1,
        "beta2": beta2,
        "DF_scaled": DF_scaled,
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
        "V_disc": 0.5 * V4,
        "T_mom": 0.5 * rho * A_annulus * V4**2,
        "CL": CL,
        "alpha": alpha,
        "B_rotor": B_rotor,
        "rho": rho,
        "P_shaft_static": 0.5 * rho * A_annulus * V4**2 * 0.5 * V4,
        "CL_target": CL_target,
        "T_bemt": T_bemt,
        "Q_bemt": Q_bemt,
        "D_profile_rel": D_prof,
        "scale_c": scale_c,
    }


def bemt_for_rotor(r, U, Vx, beta1, c, CL, rho, B, CL_target):
    # simple BEMT integration (static-like inflow)
    T = 0.0
    Q = 0.0
    D_prof_total = 0.0
    n = len(r)
    for i in range(n - 1):
        r1 = r[i]
        r2 = r[i + 1]
        dr = r2 - r1
        rm = 0.5 * (r1 + r2)
        Um = 0.5 * (U[i] + U[i + 1])
        Vxm = 0.5 * (Vx[i] + Vx[i + 1])
        cm = 0.5 * (c[i] + c[i + 1])
        CLm = 0.5 * (CL[i] + CL[i + 1])
        beta_m = 0.5 * (beta1[i] + beta1[i + 1])
        phi = math.radians(beta_m)

        W = math.sqrt(Um**2 + Vxm**2)
        # CD from polar
        CDm = cd_from_cl_single(CLm, cl_opt=CL_target, cd_min=0.015, k2=0.04)

        dL = 0.5 * rho * W**2 * cm * CLm
        dD = 0.5 * rho * W**2 * cm * CDm

        dT = B * (dL * math.cos(phi) - dD * math.sin(phi)) * dr
        dQ = B * rm * (dL * math.sin(phi) + dD * math.cos(phi)) * dr
        T += dT
        Q += dQ
        D_prof_total += B * dD * dr

    return T, Q, D_prof_total


# ============================================================
# Stator design (swirl removal + DF limit)
# ============================================================

def design_stator(rotor_res, DF_limit=0.4, B_stator=13):
    rho = rotor_res["rho"]
    r = rotor_res["r"]
    R_tip = rotor_res["R_tip"]
    Vx_in = rotor_res["Vx"]
    Vtheta_in = rotor_res["Vtheta2"]

    V1s = []
    beta1s = []
    for vx, vt in zip(Vx_in, Vtheta_in):
        V1i = math.sqrt(vx**2 + vt**2)
        V1s.append(V1i)
        beta1s.append(math.degrees(math.atan2(vx, vt)))

    Vtheta_out = [0.0 for _ in r]
    Vx_out = Vx_in[:]
    V2s = [math.sqrt(vx**2 + vt**2) for vx, vt in zip(Vx_out, Vtheta_out)]

    CL_target = 0.4
    a_stator = 0.10
    alpha0_stator = 0.0
    alpha_section = alpha_from_CL(CL_target, a_stator, alpha0_stator)
    CL = [CL_target for _ in r]
    alpha = [alpha_section for _ in r]

    c_from_DF = [0.0 for _ in r]
    DF = [0.0 for _ in r]

    for i in range(len(r)):
        ri = r[i]
        s = 2.0 * math.pi * ri / float(B_stator)
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
        if s_over_c_req > 0:
            c_from_DF[i] = s / s_over_c_req

    c_rotor = rotor_res["c"]
    c_pref = [0.8 * ci for ci in c_rotor]
    c = []
    for ci_pref, ci_df in zip(c_pref, c_from_DF):
        c.append(max(ci_pref, ci_df))

    c = smooth_list_py(c, w=0.3)

    for i in range(len(r)):
        ri = r[i]
        s = 2.0 * math.pi * ri / float(B_stator)
        V1i = V1s[i]
        V2i = V2s[i]
        Vt1 = Vtheta_in[i]
        Vt2 = Vtheta_out[i]
        ci = c[i]
        if ci <= 0.0 or V1i <= 1e-6:
            DF[i] = 0.0
            continue
        s_over_c = s / ci
        DF[i] = 1.0 - V2i / V1i + (Vt2 - Vt1) / (2.0 * V1i) * s_over_c

    # profile drag (approx)
    CD = []
    for CLi in CL:
        CD.append(cd_from_cl_single(CLi, cl_opt=0.4, cd_min=0.012, k2=0.03))
    W = V1s[:]

    D_profile = 0.0
    for i in range(len(r) - 1):
        rm = 0.5 * (r[i] + r[i + 1])
        cm = 0.5 * (c[i] + c[i + 1])
        Wm = 0.5 * (W[i] + W[i + 1])
        CDm = 0.5 * (CD[i] + CD[i + 1])
        dr = r[i + 1] - r[i]
        Dp = 0.5 * rho * Wm**2 * cm * CDm
        D_profile += Dp * dr * B_stator

    return {
        "r": r,
        "r_R": [ri / R_tip for ri in r],
        "c": c,
        "beta1": beta1s,
        "beta2": [90.0 for _ in r],
        "DF": DF,
        "CL": CL,
        "CD": CD,
        "alpha": alpha,
        "D_profile": D_profile,
        "B_stator": B_stator,
    }


# ============================================================
# Build airfoil sections in global coordinates
# ============================================================

def build_sections_with_pitch(design_res, airfoil_digits, alpha_deg, n_points=121):
    x_af, y_af = generate_naca4(airfoil_digits, n_points)
    r_list = design_res["r"]
    c_list = design_res["c"]
    beta1_list = design_res["beta1"]

    sections = []
    alpha_rad = math.radians(alpha_deg)

    for ri, ci, b1_deg in zip(r_list, c_list, beta1_list):
        beta1_rad = math.radians(b1_deg)
        beta_m = beta1_rad - alpha_rad

        x_local = []
        y_local = []
        for xn, yn in zip(x_af, y_af):
            x_local.append((xn - 0.25) * ci)  # pivot at 25% chord
            y_local.append(yn * ci)

        cosb = math.cos(beta_m)
        sinb = math.sin(beta_m)

        x_global = []
        y_global = []
        z_global = []

        for xl, yl in zip(x_local, y_local):
            s = cosb * xl - sinb * yl   # tangential
            a = sinb * xl + cosb * yl   # axial
            x_global.append(a)
            y_global.append(ri)
            z_global.append(s)

        sections.append({
            "r": ri,
            "chord": ci,
            "beta1_deg": b1_deg,
            "beta_m_deg": math.degrees(beta_m),
            "alpha_deg": alpha_deg,
            "x_global": x_global,
            "y_global": y_global,
            "z_global": z_global,
        })
    return sections


def create_blade_section_sketches(rootComp, sections, name_prefix="Rotor"):
    planes = rootComp.constructionPlanes
    sketches = rootComp.sketches

    for idx, sec in enumerate(sections):
        ri = sec["r"]
        xg = sec["x_global"]
        yg = sec["y_global"]
        zg = sec["z_global"]

        planeInput = planes.createInput()
        offsetVal = adsk.core.ValueInput.createByReal(ri)
        planeInput.setByOffset(rootComp.xZConstructionPlane, offsetVal)
        plane = planes.add(planeInput)
        plane.name = f"{name_prefix}_sec_{idx+1}_r{ri*1000.0:.1f}mm"

        sketch = sketches.add(plane)
        sketch.name = f"{name_prefix}_Section_{idx+1}"

        pts = adsk.core.ObjectCollection.create()
        for X, Y, Z in zip(xg, yg, zg):
            model_pt = adsk.core.Point3D.create(X, Y, Z)
            sk_pt = sketch.modelToSketchSpace(model_pt)
            pts.add(sk_pt)

        spline = sketch.sketchCurves.sketchFittedSplines.add(pts)
        first_pt = spline.fitPoints.item(0)
        last_pt = spline.fitPoints.item(spline.fitPoints.count - 1)
        sketch.sketchCurves.sketchLines.addByTwoPoints(last_pt, first_pt)


# ============================================================
# Fusion entry points
# ============================================================

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("No active Fusion design.", "EDF Rotor/Stator BEMT")
            return
        rootComp = design.rootComponent

        # EDF design params (same as analytical code)
        D = 0.125
        T_des = 25.0
        phi_mean = 0.75
        DF_limit = 0.4
        hub_to_tip = 0.30
        rho = 1.225
        psi_des = 0.4
        B_rotor = 11
        B_stator = 13

        rotor = design_rotor_scaled(
            D=D,
            T_des=T_des,
            phi_mean=phi_mean,
            DF_limit=DF_limit,
            hub_to_tip=hub_to_tip,
            rho=rho,
            psi_des=psi_des,
            B_rotor=B_rotor,
            n_span=16,
            cR_min=0.04,
            cR_max=0.30,
            CL_target=0.9,
        )
        stator = design_stator(rotor, DF_limit=DF_limit, B_stator=B_stator)

        # Rotor and stator AoA
        alpha_rotor = rotor["alpha"][0]
        alpha_stator = stator["alpha"][0]

        rotor_secs = build_sections_with_pitch(
            rotor,
            airfoil_digits="4412",
            alpha_deg=alpha_rotor,
            n_points=121
        )
        stator_secs = build_sections_with_pitch(
            stator,
            airfoil_digits="0012",
            alpha_deg=alpha_stator,
            n_points=121
        )

        create_blade_section_sketches(rootComp, rotor_secs, "Rotor")
        create_blade_section_sketches(rootComp, stator_secs, "Stator")

        msg = (
            "EDF rotor & stator BEMT-optimised sections created (no 'slender tip' constraint).\n"
            "Use Loft + Circular Pattern manually to form full blades & arrays.\n\n"
            f"D = {D*1000:.1f} mm, T_des ≈ {T_des:.1f} N, hub/tip = {hub_to_tip:.2f}\n"
            f"Rotor: B={B_rotor}, CL≈{rotor['CL_target']:.2f}, chord scale={rotor['scale_c']:.2f}\n"
            f"Stator: B={B_stator}, CL≈0.40 (swirl-removal)"
        )
        ui.messageBox(msg, "EDF Rotor/Stator BEMT-Optimised")

    except:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))


def stop(context):
    pass
