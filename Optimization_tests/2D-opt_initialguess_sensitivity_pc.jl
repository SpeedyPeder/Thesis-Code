using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie
using LinearAlgebra, Printf, Random, Statistics, DelimitedFiles

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
const CORIOLIS_F = 5e-2
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
const LBFGS_MAX_ITERS = parse(Int, get(ENV, "LBFGS_MAX_ITERS", "6"))
const LBFGS_G_TOL = 1e-8

const GN_MAX_ITERS = 4
const GN_G_TOL = 1e-7
const GN_DAMPING0 = 1e-6
const GN_ARMIJO_C1 = 1e-6
const GN_BACKTRACK = 0.5
const GN_MIN_STEP = 1e-3

# Output directory.
# On Windows/VSCode, set this from PowerShell with:
#   $env:SENSITIVITY_OUTPUT_DIR = "C:\\Users\\peder\\OneDrive - NTNU\\År 5\\Masteroppgave\\Optimization\\2-D_Optimization\\Sensitivity"
#
# If the environment variable is not set, results are saved in a local folder
# next to where Julia is launched.
const SAVE_DIR = get(
    ENV,
    "SENSITIVITY_OUTPUT_DIR",
    joinpath(pwd(), "sensitivity_output"),
)
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
# Optimization settings for multi-start identifiability test
# =============================================================================

# This script is meant to answer the question:
#   How sensitive is the optimization procedure to the choice of initial guess?
#
# The main idea is to solve the same calibration problem from several controlled
# initial interface guesses. The runs are compared in terms of convergence,
# final objective value, cost reduction, interface error and upper-layer velocity
# errors. This tests whether the optimization procedure is robust or sensitive
# to the starting point.

const RUN_GN_AFTER_LBFGS = parse(Bool, get(ENV, "RUN_GN_AFTER_LBFGS", "false"))
const GN_AFTER_LBFGS_ITERS = parse(Int, get(ENV, "GN_AFTER_LBFGS_ITERS", "1"))

# Set this to true if you want a more detailed L-BFGS convergence plot.
# Keeping it false avoids extra cost evaluations inside the callback and is faster
# for multi-start experiments.
const RECORD_LBFGS_EACH_ITER = false

# Keep the multi-start experiment cheap. Use a small number of deliberately
# different initial guesses, and compare the final directional velocity
# components u1 and v1 instead of only the speed.
const USE_FULL_GUESS_SET = parse(Bool, get(ENV, "USE_FULL_GUESS_SET", "true"))
const PLOT_SPEED_TOO = parse(Bool, get(ENV, "PLOT_SPEED_TOO", "false"))

# =============================================================================
# Gauss--Newton phase
# =============================================================================

function gauss_newton_phase(
    w_init;
    hist,
    max_iters=GN_MAX_ITERS,
    damping0=GN_DAMPING0,
    g_tol=GN_G_TOL,
    c1=GN_ARMIJO_C1,
    backtrack=GN_BACKTRACK,
    min_step=GN_MIN_STEP,
    name="",
)
    w = project_w_profile(copy(w_init))
    μ = damping0
    t0 = time()

    println("\nGauss--Newton phase for $name")
    println("-----------------------------")
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

    println("\nExiting Gauss--Newton phase for $name")
    println("Total GN time = $(round(time() - t0; digits=3)) seconds")

    return project_w_profile(w)
end

# =============================================================================
# Initial guesses for sensitivity experiment
# =============================================================================

function smooth_random_field(; seed=1234, amp=0.02, nsweeps=20)
    rng = MersenneTwister(seed)
    W = amp .* randn(rng, NX, NY)

    # Simple neighbor averaging to remove grid-scale noise.
    for _ in 1:nsweeps
        Wold = copy(W)
        @inbounds for j in 1:NY, i in 1:NX
            s = Wold[i, j]
            n = 1
            if i > 1
                s += Wold[i-1, j]; n += 1
            end
            if i < NX
                s += Wold[i+1, j]; n += 1
            end
            if j > 1
                s += Wold[i, j-1]; n += 1
            end
            if j < NY
                s += Wold[i, j+1]; n += 1
            end
            W[i, j] = s / n
        end
    end

    return vec(W)
end

function truth_plus_noise(; seed=1234, amp=0.005, nsweeps=20)
    perturb = smooth_random_field(seed=seed, amp=amp, nsweeps=nsweeps)
    return project_w_profile(W0_TRUE_PROFILE .+ perturb)
end

function shifted_truth_guess(; phase_shift=0.25, amp=W0_TRUE_AMP)
    w = Vector{Float64}(undef, NX * NY)

    for I in eachindex(XY_INT_REF)
        x, y = XY_INT_REF[I]
        xhat = (x - XMIN) / (XMAX - XMIN)
        yhat = (y - YMIN) / (YMAX - YMIN)
        w[I] = W0_TRUE_MEAN + amp * sin(2π * (xhat + phase_shift)) * cos(2π * yhat)
    end

    return project_w_profile(w)
end

function make_initial_guesses()
    # Controlled initial guesses for studying sensitivity to the starting point.
    # These separate errors in mean level, phase/location and perturbation size.
    return [
        ("flat_mid",       project_w_profile(fill(-0.150, NX * NY))),
        ("flat_low",       project_w_profile(fill(-0.180, NX * NY))),
        ("flat_high",      project_w_profile(fill(-0.100, NX * NY))),
        ("truth_small",    truth_plus_noise(seed=11, amp=0.003, nsweeps=20)),
        ("truth_medium",   truth_plus_noise(seed=22, amp=0.010, nsweeps=20)),
        ("truth_large",    truth_plus_noise(seed=33, amp=0.020, nsweeps=20)),
        ("shifted_truth",  shifted_truth_guess(phase_shift=0.25, amp=W0_TRUE_AMP)),
    ]
end

# =============================================================================
# One calibration run
# =============================================================================

function run_lbfgs_phase(name, w0_start, hist)
    println("\nFminbox-LBFGS phase for $name")
    println("-----------------------------")

    t_lbfgs0 = time()
    result_box = Ref{Any}()

    if RECORD_LBFGS_EACH_ITER
        time_lbfgs = @elapsed begin
            result_box[] = optimize(
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
    else
        time_lbfgs = @elapsed begin
            result_box[] = optimize(
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
                ),
            )
        end
    end

    result_lbfgs = result_box[]
    w_lbfgs = project_w_profile(copy(Optim.minimizer(result_lbfgs)))
    J_lbfgs = cost(w_lbfgs)

    push_history!(hist, "Fminbox-LBFGS-final", w_lbfgs, J_lbfgs, NaN; elapsed=time_lbfgs, force=true)

    @printf("L-BFGS complete for %-12s: J = %.12e, time = %.3f s\n", name, J_lbfgs, time_lbfgs)

    return w_lbfgs, J_lbfgs, time_lbfgs
end

function run_calibration_from_guess(name, w0_start)
    hist = History()

    J0 = cost(w0_start)
    push_history!(hist, "initial", w0_start, J0, NaN; elapsed=0.0, force=true)

    w_lbfgs, J_lbfgs, time_lbfgs = run_lbfgs_phase(name, w0_start, hist)

    if RUN_GN_AFTER_LBFGS && GN_AFTER_LBFGS_ITERS > 0
        w_final_box = Ref{Vector{Float64}}()
        time_gn = @elapsed begin
            w_final_box[] = gauss_newton_phase(
                w_lbfgs;
                hist=hist,
                max_iters=GN_AFTER_LBFGS_ITERS,
                damping0=GN_DAMPING0,
                name=name,
            )
        end
        w_final = w_final_box[]
        J_final = cost(w_final)
        push_history!(hist, "GN-final", w_final, J_final, NaN; elapsed=time_lbfgs + time_gn, force=true)
    else
        w_final = w_lbfgs
        J_final = J_lbfgs
        time_gn = 0.0
    end

    profile_error = norm(w_final .- W0_TRUE_PROFILE) / sqrt(NX * NY)

    @printf(
        "Final result for %-12s: J = %.12e, interface error = %.6e, total time = %.3f s\n",
        name,
        J_final,
        profile_error,
        time_lbfgs + time_gn,
    )

    return (;
        name,
        w0=project_w_profile(w0_start),
        w_lbfgs,
        w_final,
        J0,
        J_lbfgs,
        J_final,
        time_lbfgs,
        time_gn,
        time_total=time_lbfgs + time_gn,
        profile_error,
        hist,
    )
end

# =============================================================================
# Snapshot and comparison helpers
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
        speed=sqrt.(Array(obs.u1).^2 .+ Array(obs.v1).^2),
        label,
        t,
    )
end

function u1_rmse(a, b)
    return sqrt(mean((a.u1 .- b.u1).^2))
end

function v1_rmse(a, b)
    return sqrt(mean((a.v1 .- b.v1).^2))
end

function vector_velocity_rmse(a, b)
    return sqrt(mean((a.u1 .- b.u1).^2 .+ (a.v1 .- b.v1).^2))
end

function interface_rmse(wA, wB)
    return norm(wA .- wB) / sqrt(length(wA))
end


# =============================================================================
# Idun run/combine mode
# =============================================================================

# Running all initial guesses sequentially may take too long on Idun.
# Therefore, this file is designed for a SLURM job array:
#
#   julia --project=. 2D-opt_initialguess_idun.jl 1
#   julia --project=. 2D-opt_initialguess_idun.jl 2
#   ...
#
# or, in a job array, the script automatically reads SLURM_ARRAY_TASK_ID.
#
# After all array jobs have finished, run
#
#   julia --project=. 2D-opt_initialguess_idun.jl combine
#
# to create the pairwise comparison tables and combined plots.

const MAKE_PLOTS = parse(Bool, get(ENV, "MAKE_PLOTS", "true"))

function safe_name(name)
    return replace(String(name), r"[^A-Za-z0-9_\-]" => "_")
end

function output_prefix(name)
    return joinpath(SAVE_DIR, "2D_initialguess_sensitivity_" * safe_name(name))
end

function save_history_csv(filename, hist)
    open(filename, "w") do io
        println(io, "iter,phase,J,gnorm,time")
        for k in eachindex(hist.iter)
            @printf(
                io,
                "%d,%s,%.16e,%.16e,%.16e\n",
                hist.iter[k],
                hist.phase[k],
                hist.J[k],
                hist.gnorm[k],
                hist.time[k],
            )
        end
    end
end

function save_single_result(r)
    prefix = output_prefix(r.name)

    writedlm(prefix * "_w0.txt", r.w0)
    writedlm(prefix * "_w_lbfgs.txt", r.w_lbfgs)
    writedlm(prefix * "_w_final.txt", r.w_final)
    save_history_csv(prefix * "_history.csv", r.hist)

    open(prefix * "_summary.txt", "w") do io
        println(io, "2-D initial guess sensitivity optimization result")
        println(io, "================================================")
        println(io, "name = $(r.name)")
        println(io, "NX = $NX")
        println(io, "NY = $NY")
        println(io, "OBS_STRIDE = $OBS_STRIDE")
        println(io, "LBFGS_MAX_ITERS = $LBFGS_MAX_ITERS")
        println(io, "RUN_GN_AFTER_LBFGS = $RUN_GN_AFTER_LBFGS")
        println(io, "GN_AFTER_LBFGS_ITERS = $GN_AFTER_LBFGS_ITERS")
        println(io, "J0 = $(r.J0)")
        println(io, "J_lbfgs = $(r.J_lbfgs)")
        println(io, "J_final = $(r.J_final)")
        println(io, "relative_reduction = $((r.J0 - r.J_final) / max(r.J0, eps()))")
        println(io, "w_RMSE_true = $(r.profile_error)")
        println(io, "time_lbfgs = $(r.time_lbfgs)")
        println(io, "time_gn = $(r.time_gn)")
        println(io, "time_total = $(r.time_total)")
    end

    println("Saved single-run result files with prefix: $prefix")
end

function plot_single_result(r)
    truth_snap = snapshot(W0_TRUE_PROFILE; label="truth", t=T_END)
    final_snap = snapshot(r.w_final; label=r.name, t=T_END)

    fig = Figure(size=(1500, 900), fontsize=16)

    ax1 = Axis(fig[1, 1], title="truth: interface w")
    hm1 = heatmap!(ax1, reshape(W0_TRUE_PROFILE, NX, NY))
    Colorbar(fig[1, 2], hm1)

    ax2 = Axis(fig[1, 3], title="$(r.name): optimized interface w")
    hm2 = heatmap!(ax2, reshape(r.w_final, NX, NY))
    Colorbar(fig[1, 4], hm2)

    ax3 = Axis(fig[2, 1], title="truth: upper-layer u1")
    hm3 = heatmap!(ax3, truth_snap.u1)
    Colorbar(fig[2, 2], hm3)

    ax4 = Axis(fig[2, 3], title="$(r.name): upper-layer u1")
    hm4 = heatmap!(ax4, final_snap.u1)
    Colorbar(fig[2, 4], hm4)

    ax5 = Axis(fig[3, 1], title="truth: upper-layer v1")
    hm5 = heatmap!(ax5, truth_snap.v1)
    Colorbar(fig[3, 2], hm5)

    ax6 = Axis(fig[3, 3], title="$(r.name): upper-layer v1")
    hm6 = heatmap!(ax6, final_snap.v1)
    Colorbar(fig[3, 4], hm6)

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6)
        hidedecorations!(ax)
    end

    fig_file = output_prefix(r.name) * "_fields.png"
    save(fig_file, fig)
    println("Saved single-run field figure to: $fig_file")

    fig_conv = Figure(size=(1000, 600), fontsize=16)
    ax = Axis(
        fig_conv[1, 1],
        title="Optimization history: $(r.name)",
        xlabel="recorded iteration",
        ylabel="cost",
        yscale=log10,
    )
    lines!(ax, r.hist.iter, r.hist.J)
    scatter!(ax, r.hist.iter, r.hist.J)

    conv_file = output_prefix(r.name) * "_convergence.png"
    save(conv_file, fig_conv)
    println("Saved single-run convergence figure to: $conv_file")
end

function parse_run_indices(n)
    if length(ARGS) >= 1
        arg1 = lowercase(String(ARGS[1]))

        if arg1 == "combine"
            return :combine
        elseif arg1 == "all"
            return collect(1:n)
        elseif arg1 in ("remaining", "4-7", "last4")
            return collect(4:min(7, n))
        else
            return [parse(Int, ARGS[1])]
        end

    elseif haskey(ENV, "SLURM_ARRAY_TASK_ID")
        return [parse(Int, ENV["SLURM_ARRAY_TASK_ID"])]

    else
        # Safe default: run the first initial guess only.
        # Use "all" for a full sequential PC run.
        return [1]
    end
end

function load_result_vector(name, suffix)
    filename = output_prefix(name) * suffix
    if !isfile(filename)
        error("Missing file: $filename")
    end
    return vec(Float64.(readdlm(filename)))
end

function combine_finished_runs(initial_guesses)
    println("\n============================================================")
    println("Combining finished sensitivity runs")
    println("============================================================")

    available = Tuple{String,Vector{Float64},Vector{Float64}}[]

    for (name, _) in initial_guesses
        wfile = output_prefix(name) * "_w_final.txt"
        w0file = output_prefix(name) * "_w0.txt"

        if isfile(wfile) && isfile(w0file)
            w0 = load_result_vector(name, "_w0.txt")
            wf = load_result_vector(name, "_w_final.txt")
            push!(available, (name, w0, wf))
            println("Found result for $name")
        else
            println("Skipping $name, missing result file")
        end
    end

    if isempty(available)
        error("No finished result files found in $SAVE_DIR")
    end

    truth_snap = snapshot(W0_TRUE_PROFILE; label="truth", t=T_END)

    results = map(available) do (name, w0, wf)
        fs = snapshot(wf; label=name, t=T_END)
        J0 = cost(w0)
        Jf = cost(wf)
        werr = interface_rmse(wf, W0_TRUE_PROFILE)
        reduction = (J0 - Jf) / max(J0, eps())
        (; name, w0, w_final=wf, J0, J_final=Jf, reduction, profile_error=werr, snap=fs)
    end

    sort!(results, by = r -> r.J_final)

    summary_file = joinpath(SAVE_DIR, "2D_initialguess_sensitivity_summary_combined.txt")
    pairwise_file = joinpath(SAVE_DIR, "2D_initialguess_sensitivity_pairwise_combined.txt")

    open(summary_file, "w") do io
        println(io, "2-D initial-guess sensitivity experiment")
        println(io, "Combined after Idun job array")
        println(io, "====================================================")
        println(io, "NX = $NX, NY = $NY")
        println(io, "OBS_STRIDE = $OBS_STRIDE")
        println(io, "LBFGS_MAX_ITERS = $LBFGS_MAX_ITERS")
        println(io, "")
        @printf(io, "%-16s  %14s  %14s  %12s  %14s  %14s  %14s  %14s\n",
            "name", "J0", "J_final", "reduction", "w_RMSE_true", "u1_RMSE", "v1_RMSE", "uv_RMSE")

        for r in results
            @printf(
                io,
                "%-16s  %14.6e  %14.6e  %12.4f  %14.6e  %14.6e  %14.6e  %14.6e\n",
                r.name,
                r.J0,
                r.J_final,
                r.reduction,
                r.profile_error,
                u1_rmse(r.snap, truth_snap),
                v1_rmse(r.snap, truth_snap),
                vector_velocity_rmse(r.snap, truth_snap),
            )
        end
    end

    open(pairwise_file, "w") do io
        println(io, "Pairwise comparison of optimized interfaces and upper-layer dynamics")
        println(io, "===================================================================")
        @printf(io, "%-14s  %-14s  %16s  %16s  %16s  %16s  %16s\n",
            "run A", "run B", "interface_RMSE", "u1_RMSE", "v1_RMSE", "uv_RMSE", "abs_cost_diff")

        for ia in 1:length(results)-1
            for ib in ia+1:length(results)
                a = results[ia]
                b = results[ib]

                @printf(
                    io,
                    "%-14s  %-14s  %16.6e  %16.6e  %16.6e  %16.6e  %16.6e\n",
                    a.name,
                    b.name,
                    interface_rmse(a.w_final, b.w_final),
                    u1_rmse(a.snap, b.snap),
                    v1_rmse(a.snap, b.snap),
                    vector_velocity_rmse(a.snap, b.snap),
                    abs(a.J_final - b.J_final),
                )
            end
        end
    end

    println("Saved combined summary to:  $summary_file")
    println("Saved combined pairwise to: $pairwise_file")

    if MAKE_PLOTS
        nplots = length(results) + 1
        ncols = min(4, nplots)
        nrows = cld(nplots, ncols)

        # Combined convergence plot from saved history files.
        fig_conv = Figure(size=(1200, 700), fontsize=16)
        axc = Axis(
            fig_conv[1, 1],
            title="Sensitivity to initial guess: convergence histories",
            xlabel="recorded iteration",
            ylabel="cost",
            yscale=log10,
        )
        for r in results
            histfile = output_prefix(r.name) * "_history.csv"
            if isfile(histfile)
                lines = readlines(histfile)[2:end]
                if !isempty(lines)
                    its = Int[]
                    Js = Float64[]
                    for line in lines
                        parts = split(line, ',')
                        push!(its, parse(Int, parts[1]))
                        push!(Js, parse(Float64, parts[3]))
                    end
                    lines!(axc, its, Js, label=r.name)
                    scatter!(axc, its, Js)
                end
            end
        end
        axislegend(axc, position=:rt)
        conv_file = joinpath(SAVE_DIR, "2D_initialguess_sensitivity_convergence_combined.png")
        save(conv_file, fig_conv)
        println("Saved combined convergence figure to: $conv_file")

        fig_int = Figure(size=(1800, 900), fontsize=16)
        all_int = [("truth", W0_TRUE_PROFILE)]
        append!(all_int, [(r.name, r.w_final) for r in results])

        for (k, (name, wprofile)) in enumerate(all_int)
            row = div(k - 1, ncols) + 1
            col = mod(k - 1, ncols) + 1
            ax = Axis(fig_int[row, col], title="interface w: $name")
            hm = heatmap!(ax, reshape(wprofile, NX, NY))
            Colorbar(fig_int[row, col + ncols], hm)
            hidedecorations!(ax)
        end

        int_file = joinpath(SAVE_DIR, "2D_initialguess_sensitivity_interfaces_combined.png")
        save(int_file, fig_int)
        println("Saved combined interface figure to: $int_file")

        fig_u1 = Figure(size=(1800, 900), fontsize=16)
        all_vel = [("truth", truth_snap)]
        append!(all_vel, [(r.name, r.snap) for r in results])

        for (k, (name, snap)) in enumerate(all_vel)
            row = div(k - 1, ncols) + 1
            col = mod(k - 1, ncols) + 1
            ax = Axis(fig_u1[row, col], title="upper-layer u1: $name")
            hm = heatmap!(ax, snap.u1)
            Colorbar(fig_u1[row, col + ncols], hm)
            hidedecorations!(ax)
        end

        u1_file = joinpath(SAVE_DIR, "2D_initialguess_sensitivity_upperlayer_u1_combined.png")
        save(u1_file, fig_u1)
        println("Saved combined u1 figure to: $u1_file")

        fig_v1 = Figure(size=(1800, 900), fontsize=16)

        for (k, (name, snap)) in enumerate(all_vel)
            row = div(k - 1, ncols) + 1
            col = mod(k - 1, ncols) + 1
            ax = Axis(fig_v1[row, col], title="upper-layer v1: $name")
            hm = heatmap!(ax, snap.v1)
            Colorbar(fig_v1[row, col + ncols], hm)
            hidedecorations!(ax)
        end

        v1_file = joinpath(SAVE_DIR, "2D_initialguess_sensitivity_upperlayer_v1_combined.png")
        save(v1_file, fig_v1)
        println("Saved combined v1 figure to: $v1_file")
    end
end

# =============================================================================
# Main
# =============================================================================

println("\n============================================================")
println("2-D initial-guess sensitivity experiment")
println("============================================================")
println("SAVE_DIR = $SAVE_DIR")
println("L-BFGS iterations per guess = $LBFGS_MAX_ITERS")
println("Run GN after L-BFGS = $RUN_GN_AFTER_LBFGS")
println("GN iterations after L-BFGS = $GN_AFTER_LBFGS_ITERS")
println("Sensitivity diagnostics = convergence, final cost, interface error, u1/v1 errors")

initial_guesses = make_initial_guesses()
mode = parse_run_indices(length(initial_guesses))

if mode == :combine
    combine_finished_runs(initial_guesses)
else
    for idx in mode
        if idx < 1 || idx > length(initial_guesses)
            error("Invalid initial guess index $idx. Valid range is 1:$(length(initial_guesses))")
        end

        name, w0 = initial_guesses[idx]

        skip_existing = parse(Bool, get(ENV, "SKIP_EXISTING", "true"))
        summary_file = output_prefix(name) * "_summary.txt"

        if skip_existing && isfile(summary_file)
            println("\nSkipping initial guess $idx / $(length(initial_guesses)): $name")
            println("Existing summary found: $summary_file")
            continue
        end

        println("\n============================================================")
        println("Running initial guess $idx / $(length(initial_guesses)): $name")
        println("============================================================")

        total_time = @elapsed begin
            r = run_calibration_from_guess(name, w0)
            save_single_result(r)
            if MAKE_PLOTS
                plot_single_result(r)
            end
        end

        println("\nFinished initial guess $name")
        println("Wall time = $(round(total_time; digits=3)) s")
    end
end

println("\nDone.")