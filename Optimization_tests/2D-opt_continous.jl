using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie
using LinearAlgebra, Printf

# =============================================================================
# Constants
# =============================================================================

const NX, NY, GC = 16, 16, 2

# Physical 32 m × 32 m pool
const XMIN, XMAX = 0.0, 16
const YMIN, YMAX = 0.0, 16

const CFL_2D = 0.2
const DEPTH_CUT = 1e-4

# Rotating pool: f = 10^(-2) s^(-1)
const CORIOLIS_F = 5e-1
const ROTATION_PERIOD = 2π / CORIOLIS_F

# Simulate 3 minutes
const T_END = 3 * 60.0

const OBS_TIMES = [
    1.0 * 60.0,
    2.0 * 60.0,
    3.0 * 60.0,
]

# Interface w
const W0_INIT_CONST = -0.15
const W0_TRUE_MEAN = -0.120
const W0_TRUE_AMP = 0.025

# Initial free surface ε
const EPSILON_BACKGROUND = 0.0
const EPSILON_BUMP_AMP = 0.015
const EPSILON_BUMP_CENTER_X = 0.5 * (XMIN + XMAX)
const EPSILON_BUMP_CENTER_Y = 0.5 * (YMIN + YMAX)
const EPSILON_BUMP_RADIUS = 4.0

# Observation locations
const OBS_STRIDE = 4
const CELL_INDICES = [(i, j) for j in 1:OBS_STRIDE:NY for i in 1:OBS_STRIDE:NX]

# Objective weights
const W_U1 = 1.0
const W_V1 = 1.0

# Smoothness regularization for continuous interface
const W_REG_SMOOTH = 1e-4

# Optimization settings
const FD_CHUNK = 32

const LBFGS_M = 10
const LBFGS_MAX_ITERS = 10
const LBFGS_G_TOL = 1e-8

const GN_MAX_ITERS = 6
const GN_G_TOL = 1e-7
const GN_DAMPING0 = 1e-6
const GN_ARMIJO_C1 = 1e-6
const GN_BACKTRACK = 0.5
const GN_MIN_STEP = 1e-3

const SAVE_DIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization"
mkpath(SAVE_DIR)

# =============================================================================
# Bathymetry
# =============================================================================

function make_bottom_cos_sin_2d(; backend, grid)
    x_faces = SinFVM.cell_faces(grid, SinFVM.XDIR; interior=false)
    y_faces = SinFVM.cell_faces(grid, SinFVM.YDIR; interior=false)

    B = Matrix{Float64}(undef, length(x_faces), length(y_faces))

    @inbounds for j in eachindex(y_faces), i in eachindex(x_faces)
        x = x_faces[i]
        y = y_faces[j]

        B[i, j] =
            -0.3+
            0.04 * cos(2π * x / 16.0) +
            0.03 * sin(2π * y / 16.0)
    end

    return SinFVM.BottomTopography2D(B, backend, grid)
end

# =============================================================================
# Initial free surface ε
# =============================================================================

function epsilon0_profile(xy, ::Type{T}) where {T}
    x, y = xy

    dx = (T(x) - T(EPSILON_BUMP_CENTER_X)) / T(EPSILON_BUMP_RADIUS)
    dy = (T(y) - T(EPSILON_BUMP_CENTER_Y)) / T(EPSILON_BUMP_RADIUS)

    return T(EPSILON_BACKGROUND) + T(EPSILON_BUMP_AMP) * exp(-(dx^2 + dy^2))
end

# =============================================================================
# Bounds and true profile
# =============================================================================

function compute_physical_w_bounds_profile()
    backend = SinFVM.make_cpu_backend()

    grid = SinFVM.CartesianGrid(
        NX,
        NY;
        gc=GC,
        boundary=SinFVM.NeumannBC(),
        extent=[XMIN XMAX; YMIN YMAX],
    )

    bottom = make_bottom_cos_sin_2d(; backend, grid)

    B_int = SinFVM.collect_topography_cells(bottom, grid; interior=true)
    xy_int = SinFVM.cell_centers(grid; interior=true)

    lower = similar(B_int, Float64)
    upper = similar(B_int, Float64)

    for I in eachindex(xy_int)
        ε0 = epsilon0_profile(xy_int[I], Float64)
        lower[I] = Float64(B_int[I] + DEPTH_CUT)
        upper[I] = Float64(ε0 - DEPTH_CUT)
    end

    @assert all(lower .< upper) "Invalid physical interface bounds"

    return vec(lower), vec(upper), Array(lower), Array(upper), xy_int
end

const LOWER_W_PROFILE, UPPER_W_PROFILE, LOWER_W_FIELD, UPPER_W_FIELD, XY_INT_REF =
    compute_physical_w_bounds_profile()

project_w_profile(wvec) = clamp.(wvec, LOWER_W_PROFILE, UPPER_W_PROFILE)

function true_w_profile()
    w = Vector{Float64}(undef, NX * NY)

    for I in eachindex(XY_INT_REF)
        x, y = XY_INT_REF[I]

        xhat = (x - XMIN) / (XMAX - XMIN)
        yhat = (y - YMIN) / (YMAX - YMIN)

        w[I] =
            W0_TRUE_MEAN +
            W0_TRUE_AMP * sin(2π * xhat) * cos(2π * yhat)
    end

    return project_w_profile(w)
end

const W0_TRUE_PROFILE = true_w_profile()
const W0_INIT_PROFILE = project_w_profile(fill(W0_INIT_CONST, NX * NY))

println("Physical interface bounds:")
println("  min lower bound = $(minimum(LOWER_W_PROFILE))")
println("  max lower bound = $(maximum(LOWER_W_PROFILE))")
println("  min upper bound = $(minimum(UPPER_W_PROFILE))")
println("  max upper bound = $(maximum(UPPER_W_PROFILE))")

# =============================================================================
# Simulator
# =============================================================================

function setup_twolayer_simulator_2d(; backend=SinFVM.make_cpu_backend(), wprofile)
    T = eltype(wprofile)

    grid = SinFVM.CartesianGrid(
        NX,
        NY;
        gc=GC,
        boundary=SinFVM.NeumannBC(),
        extent=[XMIN XMAX; YMIN YMAX],
    )

    bottom = make_bottom_cos_sin_2d(; backend, grid)

    eq = SinFVM.TwoLayerShallowWaterEquations2D(
        bottom;
        ρ1=T(0.98),
        ρ2=T(1.00),
        g=T(9.81),
        depth_cutoff=T(DEPTH_CUT),
    )

    rec = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))
    flux = SinFVM.PathConservativeCentralUpwind(eq)

    cs = SinFVM.ConservedSystem(
        backend,
        rec,
        flux,
        eq,
        grid,
        [
            SinFVM.SourceTermBottom(),
            SinFVM.SourceTermNonConservative(),
            SinFVM.SourceTermCoriolis(T(CORIOLIS_F)),
        ],
    )

    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=CFL_2D)

    xy_int = SinFVM.cell_centers(grid; interior=true)
    B_int = SinFVM.collect_topography_cells(eq.B, grid; interior=true)

    initial = [begin
        ε0 = epsilon0_profile(xy_int[I], T)

        lower_w = T(B_int[I]) + T(DEPTH_CUT)
        upper_w = ε0 - T(DEPTH_CUT)

        w0 = clamp(T(wprofile[I]), lower_w, upper_w)
        h1 = ε0 - w0

        @SVector [h1, zero(T), zero(T), w0, zero(T), zero(T)]
    end for I in eachindex(xy_int)]

    SinFVM.set_current_state!(sim, initial)

    return sim, eq, grid
end

# =============================================================================
# Observables
# =============================================================================

function observable_fields(sim, eq, grid)
    st = SinFVM.current_interior_state(sim)
    Bcell = SinFVM.collect_topography_cells(eq.B, grid; interior=true)

    h1, q1, p1 = st.h1, st.q1, st.p1
    w, q2, p2 = st.w, st.q2, st.p2

    h2 = w .- Bcell
    ε = h1 .+ w

    u1 = SinFVM.desingularize.(Ref(eq), h1, q1)
    v1 = SinFVM.desingularize.(Ref(eq), h1, p1)
    u2 = SinFVM.desingularize.(Ref(eq), h2, q2)
    v2 = SinFVM.desingularize.(Ref(eq), h2, p2)

    return (; Bcell, ε, w, h1, h2, u1, v1, u2, v2)
end

# =============================================================================
# Observation callback
# =============================================================================

@with_kw mutable struct ObservableRecorder{VT,IT,OT}
    obs_times::VT
    cell_indices::IT
    next_obs::Int = 1
    data::Vector{OT} = OT[]
end

function (cb::ObservableRecorder)(time, sim)
    t = ForwardDiff.value(time)
    eq = sim.system.equation
    grid = sim.grid

    while cb.next_obs <= length(cb.obs_times) && t + 1e-12 >= cb.obs_times[cb.next_obs]
        obs = observable_fields(sim, eq, grid)

        for (i, j) in cb.cell_indices
            push!(cb.data, obs.u1[i, j])
            push!(cb.data, obs.v1[i, j])
        end

        cb.next_obs += 1
    end
end

function simulate_observations(; T_end, wprofile, obs_times, cell_indices)
    ADType = eltype(wprofile)

    sim, _, _ = setup_twolayer_simulator_2d(
        backend=SinFVM.make_cpu_backend(ADType),
        wprofile=ADType.(wprofile),
    )

    recorder = ObservableRecorder(
        obs_times=obs_times,
        cell_indices=cell_indices,
        data=ADType[],
    )

    SinFVM.simulate_to_time(sim, T_end; callback=recorder)

    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"

    return recorder.data
end

# =============================================================================
# Synthetic observations
# =============================================================================

const EXACT_OBS = simulate_observations(
    T_end=T_END,
    wprofile=W0_TRUE_PROFILE,
    obs_times=OBS_TIMES,
    cell_indices=CELL_INDICES,
)

const N_OBS_PAIRS = length(EXACT_OBS) ÷ 2

const U1_SCALE = max(maximum(abs.(EXACT_OBS[1:2:end])), 1e-2)
const V1_SCALE = max(maximum(abs.(EXACT_OBS[2:2:end])), 1e-2)

const SCALE_U1 = sqrt(W_U1 / N_OBS_PAIRS) / U1_SCALE
const SCALE_V1 = sqrt(W_V1 / N_OBS_PAIRS) / V1_SCALE

println("Generated synthetic observations:")
println("  n_controls        = $(NX * NY)")
println("  n_obs             = $(length(EXACT_OBS))")
println("  n_cells_obs       = $(length(CELL_INDICES))")
println("  domain            = [$(XMIN), $(XMAX)] × [$(YMIN), $(YMAX)] m")
println("  T_END             = $T_END s = $(T_END / 60) min")
println("  f                 = $CORIOLIS_F s⁻¹")
println("  rotation period   = $(ROTATION_PERIOD) s = $(ROTATION_PERIOD / 60) min")
println("  boundary          = Neumann")
println("  max |obs|         = $(maximum(abs.(EXACT_OBS)))")

# =============================================================================
# Cost function
# =============================================================================

function residual_vector_misfit(wvec)
    wprofile = project_w_profile(wvec)

    pred = simulate_observations(
        T_end=T_END,
        wprofile=wprofile,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    T = eltype(pred)
    nmis = length(pred)
    r = Vector{T}(undef, nmis)

    @inbounds for k in 1:2:nmis
        r[k] = T(SCALE_U1) * (pred[k] - T(EXACT_OBS[k]))
        r[k+1] = T(SCALE_V1) * (pred[k+1] - T(EXACT_OBS[k+1]))
    end

    return r
end

function residual_vector_smooth(wvec)
    T = eltype(wvec)
    W = reshape(project_w_profile(wvec), NX, NY)

    nreg_x = (NX - 1) * NY
    nreg_y = NX * (NY - 1)
    nreg = nreg_x + nreg_y

    r = Vector{T}(undef, nreg)
    scale = sqrt(T(W_REG_SMOOTH) / T(nreg))

    k = 1

    @inbounds for j in 1:NY, i in 1:NX-1
        r[k] = scale * (W[i+1, j] - W[i, j])
        k += 1
    end

    @inbounds for j in 1:NY-1, i in 1:NX
        r[k] = scale * (W[i, j+1] - W[i, j])
        k += 1
    end

    return r
end

function residual_vector(wvec)
    rmis = residual_vector_misfit(wvec)

    if W_REG_SMOOTH > 0
        return vcat(rmis, residual_vector_smooth(wvec))
    else
        return rmis
    end
end

function cost(wvec)
    r = residual_vector(wvec)
    return 0.5 * sum(abs2, r)
end

function gradient_fd(wvec)
    cfg = ForwardDiff.GradientConfig(cost, wvec, ForwardDiff.Chunk{FD_CHUNK}())
    return ForwardDiff.gradient(cost, wvec, cfg)
end

function grad!(g, wvec)
    cfg = ForwardDiff.GradientConfig(cost, wvec, ForwardDiff.Chunk{FD_CHUNK}())
    ForwardDiff.gradient!(g, cost, wvec, cfg)
    return g
end

# =============================================================================
# History
# =============================================================================

@with_kw mutable struct History
    iter::Vector{Int}=Int[]
    phase::Vector{String}=String[]
    J::Vector{Float64}=Float64[]
    gnorm::Vector{Float64}=Float64[]
    time::Vector{Float64}=Float64[]
    wprofiles::Vector{Vector{Float64}}=Vector{Vector{Float64}}()
end

function push_history!(hist, phase, wvec, J, gnorm; elapsed=NaN, force=false)
    wproj = collect(Float64.(project_w_profile(wvec)))

    if !force && !isempty(hist.wprofiles)
        diff = norm(wproj .- hist.wprofiles[end]) / sqrt(length(wproj))
        if diff < 1e-12
            return
        end
    end

    push!(hist.iter, length(hist.iter))
    push!(hist.phase, phase)
    push!(hist.J, max(Float64(J), 0.0))
    push!(hist.gnorm, Float64(gnorm))
    push!(hist.time, elapsed)
    push!(hist.wprofiles, wproj)
end

optim_x(s) = hasproperty(s, :x) ? s.x : Optim.minimizer(s)

function history_callback(hist, t0)
    return s -> begin
        w = project_w_profile(copy(optim_x(s)))

        Jphys = cost(w)
        gnorm = hasproperty(s, :g_norm) ? Float64(s.g_norm) : NaN

        push_history!(hist, "Fminbox-LBFGS", w, Jphys, gnorm; elapsed=time() - t0)

        @printf(
            "Fminbox iter %2d: J = %.12e, |g| = %.4e, time = %.3f s\n",
            length(hist.iter) - 1,
            Jphys,
            gnorm,
            time() - t0,
        )

        return false
    end
end

# =============================================================================
# Optimization
# =============================================================================

hist = History()

w0_start = copy(W0_INIT_PROFILE)

J0 = cost(w0_start)
g0 = gradient_fd(w0_start)

push_history!(hist, "initial", w0_start, J0, norm(g0); elapsed=0.0, force=true)

println("\nFminbox-LBFGS phase")
println("-------------------")
println("Iter     Function value      Gradient norm      Time")

t_lbfgs0 = time()

time_lbfgs = @elapsed begin
    global result_lbfgs = optimize(
        cost,
        grad!,
        LOWER_W_PROFILE,
        UPPER_W_PROFILE,
        w0_start,
        Fminbox(LBFGS(; m=LBFGS_M)),
        Optim.Options(
            outer_iterations=1,
            iterations=LBFGS_MAX_ITERS,
            show_trace=false,
            show_every=1,
            g_tol=LBFGS_G_TOL,
            allow_f_increases=false,
            callback=history_callback(hist, t_lbfgs0),
        ),
    )
end


function gauss_newton_phase(
    w_init;
    hist,
    max_iters=GN_MAX_ITERS,
    damping0=GN_DAMPING0,
    g_tol=GN_G_TOL,
    c1=GN_ARMIJO_C1,
    backtrack=GN_BACKTRACK,
    min_step=GN_MIN_STEP,
)
    w = project_w_profile(copy(w_init))
    μ = damping0
    t0 = time()

    println("\nGauss--Newton phase")
    println("-------------------")
    println("Iter     Function value      Gradient norm      Step length      Avg step      Time")

    for k in 1:max_iters
        r = residual_vector(w)
        Jval = 0.5 * dot(r, r)

        Jr = ForwardDiff.jacobian(residual_vector, w)

        g = Jr' * r
        gnorm = norm(g)

        if gnorm < g_tol
            push_history!(hist, "GN", w, Jval, gnorm; elapsed=time() - t0)
            println("GN converged on gradient norm.")
            break
        end

        Hgn = Jr' * Jr + μ * I
        δ = -(Hgn \ g)

        α = 1.0
        accepted = false
        w_trial = w
        J_trial = Jval

        while α >= min_step
            w_candidate = project_w_profile(w .+ α .* δ)
            r_candidate = residual_vector(w_candidate)
            J_candidate = 0.5 * dot(r_candidate, r_candidate)

            if J_candidate <= Jval + c1 * α * dot(g, δ)
                w_trial = w_candidate
                J_trial = J_candidate
                accepted = true
                break
            end

            α *= backtrack
        end

        if !accepted
            μ *= 10.0
            @printf("GN iter %2d rejected, increasing μ to %.3e\n", k, μ)
            continue
        end

        step_avg = norm(w_trial .- w) / sqrt(length(w))
        w = w_trial

        μ = J_trial < Jval ? max(0.5 * μ, 1e-12) : min(10.0 * μ, 1e6)

        push_history!(hist, "GN", w, J_trial, gnorm; elapsed=time() - t0)

        @printf(
            "%5d   %14.6e   %13.6e   %11.6f   %11.6e   %.3f s\n",
            k,
            J_trial,
            gnorm,
            α,
            step_avg,
            time() - t0,
        )
    end

    println("\nExiting Gauss--Newton phase")
    println("Total GN time = $(round(time() - t0; digits=3)) seconds")

    return project_w_profile(w)
end


w_lbfgs_profile = project_w_profile(copy(Optim.minimizer(result_lbfgs)))

J_lbfgs = cost(w_lbfgs_profile)
g_lbfgs = gradient_fd(w_lbfgs_profile)

push_history!(
    hist,
    "Fminbox-LBFGS-final",
    w_lbfgs_profile,
    J_lbfgs,
    norm(g_lbfgs);
    elapsed=time_lbfgs,
)

println("\n=== Fminbox-LBFGS complete ===")
println("Cost after L-BFGS          = $(round(J_lbfgs; digits=12))")
println("Gradient norm after L-BFGS = $(round(norm(g_lbfgs); digits=12))")
println("Time L-BFGS                = $(round(time_lbfgs; digits=4)) s")

time_gn = @elapsed begin
    global w_opt_profile = gauss_newton_phase(w_lbfgs_profile; hist=hist)
end

J_opt = cost(w_opt_profile)
g_opt = gradient_fd(w_opt_profile)

push_history!(
    hist,
    "GN-final",
    w_opt_profile,
    J_opt,
    norm(g_opt);
    elapsed=time_lbfgs + time_gn,
)

profile_error = norm(w_opt_profile .- W0_TRUE_PROFILE) / sqrt(NX * NY)

println("\n=== Optimization complete ===")
println("Final cost        = $(round(J_opt; digits=12))")
println("Final |g|         = $(round(norm(g_opt); digits=12))")
println("Profile L2 error  = $(round(profile_error; digits=12))")
println("Time L-BFGS       = $(round(time_lbfgs; digits=4)) s")
println("Time GN           = $(round(time_gn; digits=4)) s")
println("Total time        = $(round(time_lbfgs + time_gn; digits=4)) s")

# =============================================================================
# Snapshot helper
# =============================================================================

function snapshot(wprofile; label="", t=T_END)
    sim, eq, grid = setup_twolayer_simulator_2d(
        backend=SinFVM.make_cpu_backend(),
        wprofile=Float64.(wprofile),
    )

    if t > 0
        SinFVM.simulate_to_time(sim, t)
    end

    obs = observable_fields(sim, eq, grid)

    return (;
        ε=Array(obs.ε),
        w=Array(obs.w),
        B=Array(obs.Bcell),
        u1=Array(obs.u1),
        v1=Array(obs.v1),
        label,
        t,
    )
end

snaps = [
    snapshot(W0_INIT_PROFILE; label="initial", t=0.0),
    snapshot(W0_TRUE_PROFILE; label="synthetic truth", t=T_END),
    snapshot(w_opt_profile; label="optimized", t=T_END),
]

# =============================================================================
# Plotting
# =============================================================================

fig = Figure(size=(1700, 1200), fontsize=18)

ax_conv = Axis(
    fig[1, 1:6],
    title="Convergence of continuous interface optimization",
    xlabel="iteration",
    ylabel="cost",
    yscale=log10,
)

lines!(ax_conv, hist.iter, hist.J, label="J")
scatter!(ax_conv, hist.iter, hist.J)
axislegend(ax_conv, position=:rt)

for (j, s) in enumerate(snaps)
    row = j + 1
    speed_u1 = sqrt.(s.u1.^2 .+ s.v1.^2)

    ax_ε = Axis(fig[row, 1], title="$(s.label): free surface ε, t=$(round(s.t; digits=2))")
    hm_ε = heatmap!(ax_ε, s.ε)
    Colorbar(fig[row, 2], hm_ε)

    ax_w = Axis(fig[row, 3], title="$(s.label): interface w, t=$(round(s.t; digits=2))")
    hm_w = heatmap!(ax_w, s.w)
    Colorbar(fig[row, 4], hm_w)

    ax_speed = Axis(fig[row, 5], title="$(s.label): upper-layer speed, t=$(round(s.t; digits=2))")
    hm_speed = heatmap!(ax_speed, speed_u1)
    Colorbar(fig[row, 6], hm_speed)

    hidedecorations!(ax_ε)
    hidedecorations!(ax_w)
    hidedecorations!(ax_speed)
end

display(fig)

fig_file = joinpath(SAVE_DIR, "optimization_2d_continuous_interface.png")
save(fig_file, fig)

println("Saved figure to: $fig_file")