using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie
using LinearAlgebra, Printf

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
const NX, NY, GC = 32, 32, 2
const DX_TARGET = 800.0

# --------------------------------------------------------------------------
# Physical domain
# --------------------------------------------------------------------------

const XMIN, XMAX = 0.0, NX * DX_TARGET
const YMIN, YMAX = 0.0, NY * DX_TARGET

# --------------------------------------------------------------------------
# Time scales
# --------------------------------------------------------------------------

const CFL_2D = 0.2
const DEPTH_CUT = 1e-4

# Coriolis period for f = 1e-4 is about 17.5 hours,
# so simulate several hours to observe rotational effects.
const T_END = 12.0 * 3600.0

const OBS_TIMES = [
    2.0 * 3600.0,
    4.0 * 3600.0,
    6.0 * 3600.0,
    8.0 * 3600.0,
    10.0 * 3600.0,
    12.0 * 3600.0,
]

# --------------------------------------------------------------------------
# Interface parameters
# --------------------------------------------------------------------------

const W0_TRUE_CONST = 1.85
const W0_INIT_CONST = 1.00

# --------------------------------------------------------------------------
# Initial free surface ε
# --------------------------------------------------------------------------

const EPSILON_BACKGROUND = 2.80
const EPSILON_BUMP_AMP = 0.75

# Center the perturbation in the enlarged domain
const EPSILON_BUMP_CENTER_X = 0.5 * (XMIN + XMAX)
const EPSILON_BUMP_CENTER_Y = 0.5 * (YMIN + YMAX)

# Radius should scale with grid spacing/domain size
const EPSILON_BUMP_RADIUS = 4.0 * DX_TARGET

# --------------------------------------------------------------------------
# Coriolis parameter
# --------------------------------------------------------------------------

const CORIOLIS_F = 1e-4

# --------------------------------------------------------------------------
# Observation locations
# --------------------------------------------------------------------------

const CELL_INDICES = [
    (8, 8),
    (16, 16),
    (24, 24),
]

# --------------------------------------------------------------------------
# Objective weights
# --------------------------------------------------------------------------

# Include transverse velocity to capture Coriolis effects
const W_U1 = 1.0
const W_V1 = 1.0

const W_REG = 1e-10

# --------------------------------------------------------------------------
# Optimization settings
# --------------------------------------------------------------------------

const LBFGS_M = 5
const LBFGS_MAX_ITERS = 20
const LBFGS_G_TOL = 1e-8

const GN_MAX_ITERS = 10
const GN_TOL = 1e-10
const GN_MU0 = 1e-6
const GN_LINESEARCH_MAX = 12

# --------------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------------

const SAVE_DIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization"
mkpath(SAVE_DIR)

# -----------------------------------------------------------------------------
# Bathymetry
# -----------------------------------------------------------------------------

function make_bottom_cos_sin_2d(; backend, grid)
    x_faces = SinFVM.cell_faces(grid, SinFVM.XDIR; interior=false)
    y_faces = SinFVM.cell_faces(grid, SinFVM.YDIR; interior=false)
    x0, x1 = SinFVM.start_extent(grid, SinFVM.XDIR), SinFVM.end_extent(grid, SinFVM.XDIR)
    y0, y1 = SinFVM.start_extent(grid, SinFVM.YDIR), SinFVM.end_extent(grid, SinFVM.YDIR)

    Lx, Ly = x1 - x0, y1 - y0
    B = Matrix{Float64}(undef, length(x_faces), length(y_faces))
    @inbounds for j in eachindex(y_faces), i in eachindex(x_faces)
        xhat = (x_faces[i] - x0) / Lx
        yhat = (y_faces[j] - y0) / Ly

        B[i, j] =
            -3.0 +
            0.4 * cos(2π * xhat) +
            0.3 * sin(2π * yhat)
    end

    return SinFVM.BottomTopography2D(B, backend, grid)
end

# -----------------------------------------------------------------------------
# Initial free surface ε
# -----------------------------------------------------------------------------

function epsilon0_profile(xy, ::Type{T}) where {T}
    x, y = xy
    dx = (T(x) - T(EPSILON_BUMP_CENTER_X)) / T(EPSILON_BUMP_RADIUS)
    dy = (T(y) - T(EPSILON_BUMP_CENTER_Y)) / T(EPSILON_BUMP_RADIUS)
    return T(EPSILON_BACKGROUND) + T(EPSILON_BUMP_AMP) * exp(-(dx^2 + dy^2))
end

# -----------------------------------------------------------------------------
# Physical scalar bounds for constant interface w₀
# -----------------------------------------------------------------------------

function compute_physical_w_bounds()
    backend = SinFVM.make_cpu_backend()

    grid = SinFVM.CartesianGrid(
        NX,
        NY;
        gc=GC,
        boundary=SinFVM.NeumannBC(),
        extent=[XMIN XMAX; YMIN YMAX],
    )

    bottom = make_bottom_cos_sin_2d(; backend=backend, grid=grid)

    B_int = SinFVM.collect_topography_cells(bottom, grid; interior=true)
    xy_int = SinFVM.cell_centers(grid; interior=true)
    ε0_int = [epsilon0_profile(xy_int[I], Float64) for I in eachindex(xy_int)]
    w_min = maximum(B_int .+ DEPTH_CUT)
    w_max = minimum(ε0_int .- DEPTH_CUT)

    @assert w_min < w_max "No valid constant interface exists: w_min >= w_max"

    return Float64(w_min), Float64(w_max)
end

const W_MIN_PHYS, W_MAX_PHYS = compute_physical_w_bounds()

@assert W_MIN_PHYS <= W0_INIT_CONST <= W_MAX_PHYS "Initial w₀ is outside physical bounds"
@assert W_MIN_PHYS <= W0_TRUE_CONST <= W_MAX_PHYS "True w₀ is outside physical bounds"

println("Physical interface bounds:")
println("  W_MIN_PHYS = $W_MIN_PHYS")
println("  W_MAX_PHYS = $W_MAX_PHYS")

# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------

function setup_twolayer_simulator_2d(; backend=SinFVM.make_cpu_backend(), wlevel)
    T = typeof(wlevel)

    grid = SinFVM.CartesianGrid(
        NX,
        NY;
        gc=GC,
        boundary=SinFVM.NeumannBC(),
        extent=[XMIN XMAX; YMIN YMAX],
    )

    bottom = make_bottom_cos_sin_2d(; backend=backend, grid=grid)

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

        w0 = clamp(T(wlevel), lower_w, upper_w)
        h1 = ε0 - w0
        @SVector [h1, zero(T), zero(T), w0, zero(T), zero(T)]
    end for I in eachindex(xy_int)]

    SinFVM.set_current_state!(sim, initial)

    return sim, eq, grid
end

# -----------------------------------------------------------------------------
# Observables
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Observation callback
# -----------------------------------------------------------------------------

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

function simulate_observations(; T_end, wlevel, obs_times, cell_indices)
    ADType = typeof(wlevel)

    sim, _, _ = setup_twolayer_simulator_2d(
        backend=SinFVM.make_cpu_backend(ADType),
        wlevel=ADType(wlevel),
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

# -----------------------------------------------------------------------------
# Synthetic observations
# -----------------------------------------------------------------------------

const EXACT_OBS = simulate_observations(
    T_end=T_END,
    wlevel=W0_TRUE_CONST,
    obs_times=OBS_TIMES,
    cell_indices=CELL_INDICES,
)

const N_OBS_PAIRS = length(EXACT_OBS) ÷ 2

const U1_SCALE = max(maximum(abs.(EXACT_OBS[1:2:end])), 1e-2)
const V1_SCALE = max(maximum(abs.(EXACT_OBS[2:2:end])), 1e-2)

const SCALE_U1 = sqrt(W_U1 / N_OBS_PAIRS) / U1_SCALE
const SCALE_V1 = sqrt(W_V1 / N_OBS_PAIRS) / V1_SCALE

println("Generated synthetic observations:")
println("  true w₀      = $W0_TRUE_CONST")
println("  initial w₀   = $W0_INIT_CONST")
println("  n_obs        = $(length(EXACT_OBS))")
println("  observables  = u1, v1")
println("  boundary     = Neumann")

# -----------------------------------------------------------------------------
# Cost function
# -----------------------------------------------------------------------------

function residual_vector_wlevel(wlevel)
    pred = simulate_observations(
        T_end=T_END,
        wlevel=wlevel,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    T = eltype(pred)
    nmis = length(pred)
    r = Vector{T}(undef, nmis)

    @inbounds for k in 1:2:nmis
        r[k]   = T(SCALE_U1) * (pred[k]   - T(EXACT_OBS[k]))
        r[k+1] = T(SCALE_V1) * (pred[k+1] - T(EXACT_OBS[k+1]))
    end

    return r
end

function residual_vector_augmented(wlevel)
    r = residual_vector_wlevel(wlevel)
    T = eltype(r)

    if W_REG > 0
        return vcat(r, sqrt(T(W_REG)) * (T(wlevel) - T(W0_INIT_CONST)))
    else
        return r
    end
end

raw_cost(wlevel) = 0.5 * sum(x -> x^2, residual_vector_wlevel(wlevel))
cost(wvec) = 0.5 * sum(x -> x^2, residual_vector_augmented(wvec[1]))

grad!(g, wvec) = ForwardDiff.gradient!(g, cost, wvec)



# -----------------------------------------------------------------------------
# History tracking
# -----------------------------------------------------------------------------
@with_kw mutable struct History
    iter::Vector{Int}=Int[]
    phase::Vector{String}=String[]
    w0::Vector{Float64}=Float64[]
    J::Vector{Float64}=Float64[]
    gnorm::Vector{Float64}=Float64[]
    time::Vector{Float64}=Float64[]
end

function push_history!(hist, phase, w0, J, gnorm; elapsed=NaN, force=false)
    w = Float64(w0)

    if !force && !isempty(hist.w0) && isapprox(w, hist.w0[end]; atol=1e-13, rtol=1e-13)
        return
    end

    push!(hist.iter, length(hist.iter))
    push!(hist.phase, phase)
    push!(hist.w0, w)
    push!(hist.J, max(Float64(J), 0.0))
    push!(hist.gnorm, Float64(gnorm))
    push!(hist.time, elapsed)
end

optim_x(s) = hasproperty(s, :x) ? s.x : Optim.minimizer(s)

function history_callback(hist, t0)
    return s -> begin
        x = optim_x(s)
        w = Float64(x[1])

        Jphys = cost([w])
        g = ForwardDiff.gradient(cost, [w])
        gnorm = norm(g)

        push_history!(hist, "Fminbox-LBFGS", w, Jphys, gnorm; elapsed=time() - t0)

        @printf(
            "Fminbox iter %2d: w = %.10f, J = %.12e, |g| = %.4e, time = %.3f s\n",
            length(hist.iter)-1,
            w,
            Jphys,
            gnorm,
            time() - t0,
        )

        return false
    end
end

# -----------------------------------------------------------------------------
# Phase 1: Fminbox + L-BFGS
# -----------------------------------------------------------------------------
hist = History()

J0 = cost([W0_INIT_CONST])
g0 = ForwardDiff.gradient(cost, [W0_INIT_CONST])
push_history!(hist, "initial", W0_INIT_CONST, J0, norm(g0); elapsed=0.0, force=true)

println("\nFminbox-LBFGS phase")
println("-------------------")
println("Iter     w₀           Function value      Gradient norm      Time")

t_lbfgs0 = time()

time_lbfgs = @elapsed begin
    global result_lbfgs = optimize(
        cost,
        grad!,
        [W_MIN_PHYS],
        [W_MAX_PHYS],
        [W0_INIT_CONST],
        Fminbox(LBFGS(; m=LBFGS_M)),
        Optim.Options(
            outer_iterations = 1,
            iterations = LBFGS_MAX_ITERS,
            show_trace = false,   # hide Optim's internal barrier values
            show_every = 1,
            g_tol = LBFGS_G_TOL,
            allow_f_increases = false,
            callback = history_callback(hist, t_lbfgs0),
        ),
    )
end

w_lbfgs = Optim.minimizer(result_lbfgs)[1]
J_lbfgs = cost([w_lbfgs])
g_lbfgs = ForwardDiff.gradient(cost, [w_lbfgs])

push_history!(
    hist,
    "Fminbox-LBFGS-final",
    w_lbfgs,
    J_lbfgs,
    norm(g_lbfgs);
    elapsed=time_lbfgs,
)

println("\n=== Fminbox-LBFGS complete ===")
println("w after Fminbox-LBFGS = $(round(w_lbfgs; digits=10))")
println("physical cost         = $(round(J_lbfgs; digits=12))")
println("gradient norm         = $(round(norm(g_lbfgs); digits=12))")
println("time Fminbox-LBFGS    = $(round(time_lbfgs; digits=4)) s")

# -----------------------------------------------------------------------------
# Phase 2: bounded Gauss--Newton refinement
# -----------------------------------------------------------------------------
function gauss_newton_bounded(w_start; hist=nothing)
    w = clamp(Float64(w_start), W_MIN_PHYS, W_MAX_PHYS)
    μ = GN_MU0
    t0 = time()

    println("\nGauss--Newton phase")
    println("-------------------")
    println("Iter     w₀           Function value      Gradient norm      Step length      Time")

    for it in 1:GN_MAX_ITERS
        iter_time = @elapsed begin
            r = residual_vector_augmented(w)

            Jmat = ForwardDiff.jacobian(z -> residual_vector_augmented(z[1]), [w])
            Jr = vec(Jmat)

            g = dot(Jr, r)
            H = dot(Jr, Jr) + μ

            step = -g / H

            if abs(step) < GN_TOL
                println("GN stopped: small step at iteration $it")
                return w
            end

            J_old = cost([w])
            accepted = false
            α = 1.0
            w_trial = w

            for _ in 1:GN_LINESEARCH_MAX
                w_candidate = clamp(w + α * step, W_MIN_PHYS, W_MAX_PHYS)
                J_new = cost([w_candidate])

                if J_new <= J_old
                    w_trial = w_candidate
                    accepted = true
                    break
                end

                α *= 0.5
            end

            if !accepted
                μ *= 10.0
                @printf("GN iter %2d rejected, increasing μ to %.3e\n", it, μ)
                continue
            end

            w = w_trial
            μ = max(μ / 2.0, 1e-12)
        end

        J_current = cost([w])
        g_current = ForwardDiff.gradient(cost, [w])
        gnorm = norm(g_current)

        push_history!(
            hist,
            "GN",
            w,
            J_current,
            gnorm;
            elapsed=time() - t0,
        )

        @printf(
            "%5d   %.10f   %14.6e   %13.6e   %11.6f   %.3f s\n",
            it,
            w,
            J_current,
            gnorm,
            α,
            time() - t0,
        )

        if gnorm < GN_TOL
            println("GN stopped: small gradient at iteration $it")
            break
        end
    end

    println("\nExiting Gauss--Newton phase")
    println("Total GN time = $(round(time() - t0; digits=3)) seconds")

    return w
end
time_gn = @elapsed begin
    global w_opt = gauss_newton_bounded(w_lbfgs; hist=hist)
end

J_opt = cost([w_opt])
g_opt = ForwardDiff.gradient(cost, [w_opt])

push_history!(
    hist,
    "bounded-GN-final",
    w_opt,
    J_opt,
    norm(g_opt);
    elapsed=time_lbfgs + time_gn,
)

println("\n=== Optimization complete ===")
println("Bounds       = [$W_MIN_PHYS, $W_MAX_PHYS]")
println("True w₀      = $W0_TRUE_CONST")
println("LBFGS w₀     = $(round(w_lbfgs; digits=10))")
println("Recovered w₀ = $(round(w_opt; digits=10))")
println("Abs. error   = $(round(abs(w_opt - W0_TRUE_CONST); digits=12))")
println("Final cost   = $(round(J_opt; digits=12))")
println("Final |g|    = $(round(norm(g_opt); digits=12))")
println("Time LBFGS   = $(round(time_lbfgs; digits=4)) s")
println("Time GN      = $(round(time_gn; digits=4)) s")
println("Total time   = $(round(time_lbfgs + time_gn; digits=4)) s")

# -----------------------------------------------------------------------------
# Snapshot helper
# -----------------------------------------------------------------------------

function snapshot(wlevel; label="", t=T_END)
    sim, eq, grid = setup_twolayer_simulator_2d(
        backend=SinFVM.make_cpu_backend(),
        wlevel=Float64(wlevel),
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
    snapshot(W0_INIT_CONST; label="initial", t=0.0),
    snapshot(W0_TRUE_CONST; label="synthetic truth", t=T_END),
    snapshot(w_opt; label="optimized", t=T_END),
]

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

fig = Figure(size=(1700, 1200), fontsize=18)

ax_conv = Axis(
    fig[1, 1:6],
    title="Convergence of interface parameter w₀",
    xlabel="iteration",
    ylabel="w₀",
)

lines!(ax_conv, hist.iter, hist.w0, label="recovered w₀")
scatter!(ax_conv, hist.iter, hist.w0)
hlines!(ax_conv, [W0_TRUE_CONST], linestyle=:dash, label="true w₀")
axislegend(ax_conv, position=:rb)

for (j, s) in enumerate(snaps)
    row = j + 1

    speed_u1 = sqrt.(s.u1.^2 .+ s.v1.^2)

    ax_ε = Axis(fig[row, 1], title="$(s.label): free surface ε, t=$(s.t)")
    hm_ε = heatmap!(ax_ε, s.ε)
    Colorbar(fig[row, 2], hm_ε)

    ax_w = Axis(fig[row, 3], title="$(s.label): interface w, t=$(s.t)")
    hm_w = heatmap!(ax_w, s.w)
    Colorbar(fig[row, 4], hm_w)

    ax_speed = Axis(fig[row, 5], title="$(s.label): upper-layer speed, t=$(s.t)")
    hm_speed = heatmap!(ax_speed, speed_u1)
    Colorbar(fig[row, 6], hm_speed)

    hidedecorations!(ax_ε)
    hidedecorations!(ax_w)
    hidedecorations!(ax_speed)
end

display(fig)

fig_file = joinpath(SAVE_DIR, "optimization_2d_constant_interface.png")
save(fig_file, fig)

println("Saved figure to: $fig_file")