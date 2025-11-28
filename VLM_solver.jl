module LinearRotorVLM

using StaticArrays, LinearAlgebra
using Printf

"Configuration for operating conditions."
struct OperatingPoint
    Vinf::SVector{3,Float64}   # freestream velocity (global frame)
    Omega::SVector{3,Float64}  # rotor angular velocity vector
    rho::Float64               # air density
end

"Geometric definition of a blade section (from CAD / parametric)."
struct BladeSectionSpec
    r::Float64       # radius at section reference (e.g. quarter-chord)
    chord::Float64   # chord length
    twist::Float64   # pitch angle (rad) at this section
end

"One bound vortex panel on the blade."
struct Panel
    p1::SVector{3,Float64}
    p2::SVector{3,Float64}
    p3::SVector{3,Float64}
    p4::SVector{3,Float64}
    cp::SVector{3,Float64}       # control point
    n_hat::SVector{3,Float64}    # unit normal
    gamma_A::SVector{3,Float64}  # bound vortex start
    gamma_B::SVector{3,Float64}  # bound vortex end
end

"One blade made of panels, at a given azimuth."
struct Blade
    panels::Vector{Panel}
    psi::Float64                 # azimuth angle (rad)
end

"Rotor geometry and discretization."
struct Rotor
    blade::Blade
    B::Int
    hub_center::SVector{3,Float64}
    R::Float64                   # tip radius
end

"Solution of the VLM system."
struct VLMSolution
    Γ::Vector{Float64}           # vortex strengths, one per panel
    thrust::Float64
    torque::Float64
    power::Float64
end


function make_sections(rhat::Vector{Float64},
                       chord::Vector{Float64},
                       pitch_deg::Vector{Float64},
                       Rtip::Float64)
    sections = BladeSectionSpec[]
    for i in eachindex(rhat)
        r_abs   = rhat[i] * Rtip
        c_abs   = chord[i] * Rtip
        twist   = deg2rad( pitch_deg[i])   # assuming your table is in deg
        push!(sections, BladeSectionSpec(r_abs, c_abs, twist))
    end
    sort!(sections, by = s -> s.r)
    return sections
end

# Small helper: rotate vector v about unit axis k by angle θ
function rotate_about_axis(v::SVector{3,Float64},
                           k::SVector{3,Float64},
                           θ::Float64)
    c, s = cos(θ), sin(θ)
    return c*v + s*(cross(k, v)) + (1-c)*(dot(k, v))*k
end

"""
    build_blade(sections, hub_center, psi; nc=4)

Build a Blade from BladeSectionSpec (sorted in increasing r).

Spacing:
- Spanwise: uniform between consecutive section radii (you control where sections are).
- Chordwise: cosine spacing between LE and TE for nc panels, clustering near LE/TE.

Geometry:
- Rotor axis = +z.
- Radial direction e_r at azimuth psi.
- Leading edge lies along radial line from hub_center.
- Chord direction obtained by rotating +z (axial) about e_r by `twist` (geometric pitch).
  Positive twist rotates TE in direction of rotation (e_t).

Control point: 3/4-chord, mid-span.
Bound vortex: 1/4-chord, from inner to outer edge.
"""
function build_blade(sections::Vector{BladeSectionSpec},
                     hub_center::SVector{3,Float64},
                     psi::Float64;
                     nc::Int = 4)::Blade

    @assert length(sections) ≥ 2 "Need at least two sections in radius."

    # Rotor-frame unit vectors for this blade azimuth
    e_r = @SVector [cos(psi), sin(psi), 0.0]      # radial outwards
    e_t = @SVector [-sin(psi), cos(psi), 0.0]     # tangential (direction of rotation)
    e_z = @SVector [0.0, 0.0, 1.0]                # rotor axis / axial

    panels = Panel[]

    # --- Spanwise: uniform between given section radii ---
    # For each adjacent pair of sections, create one "strip" of nc chordwise panels.
    for i in 1:(length(sections)-1)
        s_in  = sections[i]
        s_out = sections[i+1]

        r_in,  c_in,  β_in  = s_in.r,  s_in.chord,  s_in.twist
        r_out, c_out, β_out = s_out.r, s_out.chord, s_out.twist

        # Leading edge points at inner/outer radius (pitch axis along LE)
        LE_in  = hub_center .+ r_in  .* e_r
        LE_out = hub_center .+ r_out .* e_r

        # Local chord directions: start from +z and rotate about e_r by twist angle
        # This gives a chord lying in the tangential-axial plane.
        c_hat_in  = cos(β_in)  * e_t + sin(β_in)  * e_z
        c_hat_out = cos(β_out) * e_t + sin(β_out) * e_z

        # Trailing edge reference points
        TE_in  = LE_in  .+ c_in  .* c_hat_in
        TE_out = LE_out .+ c_out .* c_hat_out

        # --- Chordwise: cosine spacing from LE (η=0) to TE (η=1) ---
        # η_k = 0.5 * (1 - cos(π k / nc)), k=0..nc  (clustered near 0 and 1)
        η_nodes = [0.5 * (1 - cos(pi*k/nc)) for k in 0:nc]

        for j in 1:nc
            η0 = η_nodes[j-1+1]      # η_nodes[j] in 1-based indexing
            η1 = η_nodes[j+1]        # next node

            # Four corners: p1 (inner, η0), p2 (outer, η0), p3 (outer, η1), p4 (inner, η1)
            p1 = LE_in  .+ η0 * (TE_in  .- LE_in)
            p2 = LE_out .+ η0 * (TE_out .- LE_out)
            p3 = LE_out .+ η1 * (TE_out .- LE_out)
            p4 = LE_in  .+ η1 * (TE_in  .- LE_in)

            # Control point: 3/4 chord, mid-span between inner and outer
            η_cp = 0.75
            cp_in  = LE_in  .+ η_cp * (TE_in  .- LE_in)
            cp_out = LE_out .+ η_cp * (TE_out .- LE_out)
            cp = 0.5 .* (cp_in .+ cp_out)

            # Bound vortex line: quarter-chord between inner and outer
            η_gamma = 0.25
            gA = LE_in  .+ η_gamma * (TE_in  .- LE_in)
            gB = LE_out .+ η_gamma * (TE_out .- LE_out)

            # Panel normal
            v1 = p2 - p1
            v2 = p4 - p1
            n_raw = cross(v1, v2)
            n_hat = n_raw / norm(n_raw)

            push!(panels, Panel(SVector{3,Float64}(p1),
                                SVector{3,Float64}(p2),
                                SVector{3,Float64}(p3),
                                SVector{3,Float64}(p4),
                                SVector{3,Float64}(cp),
                                n_hat,
                                SVector{3,Float64}(gA),
                                SVector{3,Float64}(gB)))
        end
    end

    return Blade(panels, psi)
end


"""
    make_rotor(blade::Blade, B::Int, hub_center::SVector{3,Float64}, Rtip::Float64) -> Rotor

Convenience constructor for Rotor.
"""
make_rotor(blade::Blade, B::Int,
           hub_center::SVector{3,Float64},
           Rtip::Float64) = Rotor(blade, B, hub_center, Rtip)
"Biot–Savart induced velocity from a finite vortex segment A→B at point P."

function induced_velocity_segment(P::SVector{3,Float64},
                                  A::SVector{3,Float64},
                                  B::SVector{3,Float64};
                                  core_radius::Float64=1e-4)::SVector{3,Float64}
   	r1 = P - A
	r2 = P - B
	r0 = B - A
	n1 = norm(r1); n2 = norm(r2); n0 = norm(r0)
# Avoid singularities exactly on endpoints/line
	if n1 < 1e-12 || n2 < 1e-12 || n0 < 1e-15
    	return SVector(0.0, 0.0, 0.0)
	end

	c12 = cross(r1, r2)
	c12_2 = dot(c12, c12)

	term = dot(r0, (r1 / n1 - r2 / n2))
	denom = 4π * (c12_2 + (core_radius^2) * (n0^2))

	return (c12 * (term / denom))

end

"""
    induced_velocity_horseshoe(P, A, B, W1, W2; core_radius=1e-4)

Induced velocity at point P from a unit-strength horseshoe vortex:
- Bound segment from A to B
- Wake leg 1 from B to W1
- Wake leg 2 from W2 to A

All returns are per unit circulation Γ (multiply by Γ_j externally).
"""
function induced_velocity_horseshoe(P::SVector{3,Float64},
                                    A::SVector{3,Float64},
                                    B::SVector{3,Float64},
                                    W1::SVector{3,Float64},
                                    W2::SVector{3,Float64};
                                    core_radius::Float64 = 1e-4
                                   )::SVector{3,Float64}

    u_bound = induced_velocity_segment(P, A, B; core_radius=core_radius)

    # Wake legs: typically extend downstream from B and A
    u_wake1 = induced_velocity_segment(P, B, W1; core_radius=core_radius)
    u_wake2 = induced_velocity_segment(P, W2, A; core_radius=core_radius)

    return u_bound + u_wake1 + u_wake2
end

"""
    assemble_system(blade::Blade, op::OperatingPoint;
                    wake_length::Float64 = 4.0)

Assemble the VLM linear system A*Γ = b for a single blade.
- A_ij: normal velocity at control point i induced by unit-strength
        horseshoe vortex on panel j.
- b_i:  - (normal component of rigid-body velocity) at control point i.

Returns (A, b, panels_flat, wake_endpoints),
where:
- A::Matrix{Float64}  size Np x Np
- b::Vector{Float64}  length Np
- panels_flat::Vector{Panel}  == blade.panels (for convenience)
- wake_endpoints::Vector{Tuple{SVector{3,Float64},SVector{3,Float64}}}
    giving (W1_j, W2_j) for each panel j.
"""
function assemble_system(blade::Blade,
                         op::OperatingPoint;
                         wake_length::Float64 = 4.0)

    panels = blade.panels
    Np = length(panels)

    A = zeros(Float64, Np, Np)
    b = zeros(Float64, Np)

    # Precompute wake endpoints for each bound vortex segment
    # Simple model: wake legs aligned with local convection velocity
    wake_endpoints = Vector{Tuple{SVector{3,Float64},SVector{3,Float64}}}(undef, Np)

    for j in 1:Np
        panel = panels[j]
        A_j = panel.gamma_A
        B_j = panel.gamma_B

        # Take TE midpoint as starting point for wake (you can also use B_j / A_j)
        TE_mid = 0.5 .* (panel.p3 + panel.p4)

        # Local convection velocity: freestream + rotation
        V_conv = op.Vinf + cross(op.Omega, TE_mid)

        # Normalize and extend downstream by wake_length * Rtip-equivalent
        Vmag = norm(V_conv)
        if Vmag < 1e-8
            # Fallback: use +z as wake direction if convection is nearly zero
            e_wake = SVector(0.0, 0.0, 1.0)
        else
            e_wake = V_conv / Vmag
        end

        W1 = B_j + wake_length * e_wake   # from B → downstream
        W2 = A_j + wake_length * e_wake   # from A → downstream

        wake_endpoints[j] = (W1, W2)
    end

    # Build influence matrix A
    for i in 1:Np
        pi = panels[i]
        cp_i = pi.cp
        n_i  = pi.n_hat

        for j in 1:Np
            pj = panels[j]
            A_j = pj.gamma_A
            B_j = pj.gamma_B
            (W1_j, W2_j) = wake_endpoints[j]

            # Induced velocity at cp_i from unit-strength horseshoe on panel j
            u_ij = induced_velocity_horseshoe(cp_i, A_j, B_j, W1_j, W2_j)

            # Influence coefficient: normal component per unit Γ_j
            A[i, j] = dot(n_i, u_ij)
        end
    end

    # Build RHS b: -n · V_rigid at each control point
    for i in 1:Np
        pi = panels[i]
        cp_i = pi.cp
        n_i  = pi.n_hat

        V_rigid = op.Vinf + cross(op.Omega, cp_i)
        b[i] = -dot(n_i, V_rigid)
    end

    return A, b, panels, wake_endpoints
end

"""
    compute_thrust(blade::Blade, op::OperatingPoint, Γ::Vector{Float64};
                   B::Int = 1) -> thrust_total

Compute rotor thrust (positive along +z) from bound panel circulations Γ
for a single representative blade, then multiply by B blades.

- Uses Kutta–Joukowski: dF = ρ Γ (V_rel × t̂) * span_length
- V_rel is local relative velocity at the bound-vortex midpoint:
  V_rel = V_inf + Ω × r_mid   (induced velocity neglected at first pass).
"""
function compute_thrust(blade::Blade,
                        op::OperatingPoint,
                        Γ::Vector{Float64};
                        B::Int = 1)

    panels = blade.panels
    Np = length(panels)
    @assert length(Γ) == Np "Γ length must match number of panels"
    
    torque_per_blade = 0.0
    thrust_per_blade = 0.0

    for j in 1:Np
        panel = panels[j]
        gamma = Γ[j]

        # Bound vortex segment endpoints and direction
        A_j = panel.gamma_A
        B_j = panel.gamma_B
        r_mid = 0.5 .* (A_j + B_j)
        t_vec = B_j - A_j
        span_len = norm(t_vec)
        if span_len < 1e-12 || abs(gamma) < 1e-12
            continue
        end
        t_hat = t_vec / span_len

        # Local relative velocity at bound segment midpoint (no induced yet)
        V_rel = op.Vinf - cross(op.Omega, r_mid)

        # Kutta–Joukowski force per unit span: ρ Γ (V_rel × t̂)
        dF_per_span = op.rho * gamma * cross(V_rel, t_hat)

        # Force on this panel (approx): span_len * dF_per_span
        dF = span_len .* dF_per_span

        # Add axial component (z) to thrust per blade
        thrust_per_blade += dF[3]

	dτ = cross(r_mid, dF)
        torque_per_blade += dτ[3]
    end

    # Total rotor thrust = B * per-blade thrust
    return B * thrust_per_blade, B*torque_per_blade
end

function cl_lookup(alpha_deg::Float64,
                   alpha_tab::Vector{Float64},
                   cl_tab::Vector{Float64})
    n = length(alpha_tab)
    if alpha_deg <= alpha_tab[1]
        return cl_tab[1]
    elseif alpha_deg >= alpha_tab[end]
        return cl_tab[end]
    end
    for k in 1:n-1
        a1 = alpha_tab[k]
        a2 = alpha_tab[k+1]
        if alpha_deg >= a1 && alpha_deg <= a2
            t = (alpha_deg - a1)/(a2 - a1)
            return cl_tab[k]*(1 - t) + cl_tab[k+1]*t
        end
    end
    return cl_tab[end]
end


function solve_nonlinear_vlm_with_polars!(gamma::Vector{Float64},
                                          blade::Blade,
                                          op::OperatingPoint,
                                          rhat::Vector{Float64},
                                          chord::Vector{Float64},
                                          pitch_deg::Vector{Float64},
                                          Rtip::Float64,
                                          wake_endpoints;
                                          alpha_tab::Vector{Float64},
                                          cl_tab::Vector{Float64},
                                          nc::Int,
                                          max_iter::Int = 30,
                                          relax::Float64 = 0.1)

    panels = blade.panels
    Np = length(panels)
    Ns = length(rhat) - 1
    @assert length(gamma) == Np

    # total velocity at panel control points
    Vtot = [SVector{3,Float64}(0.0, 0.0, 0.0) for _ in 1:Np]
    gamma_new = similar(gamma)
    fill!(gamma_new, 0.0)   # avoid uninitialized junk

    # --- initial guess: rigid AoA + polar CL, uniform over chord in each strip ---
    for k in 1:Ns
        r_in  = rhat[k]   * Rtip
        r_out = rhat[k+1] * Rtip
        r_mid = 0.5 * (r_in + r_out)

        c_k        = chord[k]*Rtip
        beta_k_deg = pitch_deg[k]
        beta_k     = deg2rad(beta_k_deg)
	#
	        Vax_sum  = 0.0
        Vtan_sum = 0.0

        for j in 1:nc
            idx = (k-1)*nc + j
            cp  = panels[idx].cp
            V   = Vtot[idx]

            r_xy = sqrt(cp[1]^2 + cp[2]^2) + 1e-16
            e_t  = SVector(-cp[2]/r_xy, cp[1]/r_xy, 0.0)

            Vtan_sum += dot(V, e_t)
            Vax_sum  += -V[3]
        end

        Vtan = Vtan_sum / nc
        Vax  = Vax_sum  / nc
        V_rel = sqrt(Vtan^2 + Vax^2)
        phi   = atan(Vax , Vtan)
       alpha_deg = rad2deg(beta_k - phi)
        CL_k = cl_lookup(alpha_deg, alpha_tab, cl_tab)

        gamma_strip = 0.5 * V_rel * c_k * CL_k

        for j in 1:nc
            idx = (k-1)*nc + j
            gamma[idx] = gamma_strip / nc
        end
    end

    # --- nonlinear iterations ---
    for it in 1:max_iter

        # 1) total velocity at each control point
        for i in 1:Np
            pi = panels[i]
            cp = pi.cp
            Vind = SVector(0.0, 0.0, 0.0)

            for j in 1:Np
                pj = panels[j]
                A_j = pj.gamma_A
                B_j = pj.gamma_B
                W1_j, W2_j = wake_endpoints[j]

                u_ij = induced_velocity_horseshoe(cp, A_j, B_j, W1_j, W2_j)
                Vind += gamma[j] * u_ij
            end

            Vrigid = op.Vinf + cross(op.Omega, cp)
            Vtot[i] = Vrigid + Vind
        end

        # 2) update gamma per strip from polars using new Vtot
        for k in 1:Ns
            r_in  = rhat[k]   * Rtip
            r_out = rhat[k+1] * Rtip
            r_mid = 0.5 * (r_in + r_out)

            c_k        = chord[k]*Rtip
            beta_k_deg = pitch_deg[k]
            beta_k     = deg2rad(beta_k_deg)

            Vax_sum  = 0.0
            Vtan_sum = 0.0

            for j in 1:nc
                idx = (k-1)*nc + j
                cp  = panels[idx].cp
                V   = Vtot[idx]

                r_xy = sqrt(cp[1]^2 + cp[2]^2) + 1e-16
                e_t  = SVector(-cp[2]/r_xy, cp[1]/r_xy, 0.0)

                Vtan_sum += dot(V, e_t)
                Vax_sum  += -V[3]
            end
	    Vtan = Vtan_sum / nc
	    Vax  = Vax_sum  / nc

	    V_rel = sqrt(Vtan^2 + Vax^2)
	    phi   = atan(Vax, Vtan)
       	    alpha_deg = rad2deg(beta_k - phi)
            CL_k = cl_lookup(alpha_deg, alpha_tab, cl_tab)

            gamma_strip = 0.5 * V_rel * c_k * CL_k

            for j in 1:nc
                idx = (k-1)*nc + j
                gamma_target = gamma_strip / nc
                gamma_new[idx] = (1 - relax) * gamma[idx] + relax * gamma_target
            end
        end

        gamma .= gamma_new
    end

    return gamma
end


# Simple 1D linear interpolation, same behavior as cl_lookup
function interp1(x_tab::Vector{Float64}, y_tab::Vector{Float64}, x::Float64)
    n = length(x_tab)
    if x <= x_tab[1]
        return y_tab[1]
    elseif x >= x_tab[end]
        return y_tab[end]
    end
    for k in 1:n-1
        x1 = x_tab[k]; x2 = x_tab[k+1]
        if x >= x1 && x <= x2
            t = (x - x1)/(x2 - x1)
            return y_tab[k]*(1-t) + y_tab[k+1]*t
        end
    end
    return y_tab[end]
end

function cd_lookup(alpha_deg::Float64,
                   alpha_tab::Vector{Float64},
                   cd_tab::Vector{Float64})
    return interp1(alpha_tab, cd_tab, alpha_deg)
end


"""
    extend_polar_viterna(alpha_base_deg, cl_base, cd_base;
                         alpha_stall_deg, cd_max, alpha_max_deg=90.0)

Extend an airfoil polar from ±alpha_stall_deg out to ±alpha_max_deg
using the Viterna–Corrigan post-stall model. Angles in degrees.

Inputs:
  alpha_base_deg, cl_base, cd_base : measured/XFOIL data up to stall
  alpha_stall_deg                  : chosen stall angle (deg)
  cd_max                           : drag coefficient at 90° (flat-plate-like)
Outputs:
  alpha_ext_deg, cl_ext, cd_ext    : extended polar tables from -alpha_max to +alpha_max
"""
function extend_polar_viterna(alpha_base_deg::Vector{Float64},
                              cl_base::Vector{Float64},
                              cd_base::Vector{Float64};
                              alpha_stall_deg::Float64,
                              cd_max::Float64,
                              alpha_max_deg::Float64 = 90.0)

    # CL_s, CD_s at stall angle
    CL_s = interp1(alpha_base_deg, cl_base, alpha_stall_deg)
    CD_s = interp1(alpha_base_deg, cd_base, alpha_stall_deg)
    αs   = deg2rad(alpha_stall_deg)

    # Viterna–Corrigan coefficients (one common variant)
    B1 = cd_max
    B2 = (CD_s - B1 * sin(αs)^2) / cos(αs)

    A1 = 0.5 * B1
    # avoid division by zero if cos(αs) is tiny
    A2 = (CL_s - A1 * sin(2*αs)) * sin(αs) / max(cos(αs)^2, 1e-6)

    # Build a dense symmetric alpha grid, e.g. -90:1:90
    alpha_ext_deg = collect(-alpha_max_deg:1.0:alpha_max_deg)
    n_ext = length(alpha_ext_deg)
    cl_ext = zeros(Float64, n_ext)
    cd_ext = zeros(Float64, n_ext)

    for (i, a_deg) in pairs(alpha_ext_deg)
        a = deg2rad(a_deg)

        if abs(a_deg) <= alpha_stall_deg
            # Use original data (interpolated) in attached-flow region
            cl_ext[i] = interp1(alpha_base_deg, cl_base, a_deg)
            cd_ext[i] = interp1(alpha_base_deg, cd_base, a_deg)
        else
            # Viterna–Corrigan post-stall
            s   = sin(a); c = cos(a)
            s2  = sin(2a)
            s   = sign(s)*max(abs(s), 1e-6)  # avoid division by zero

            cl_ext[i] = A1 * s2 + A2 * (c^2 / s)
            cd_ext[i] = B1 * (s^2) + B2 * c
        end
    end

    return alpha_ext_deg, cl_ext, cd_ext
end


end # module
