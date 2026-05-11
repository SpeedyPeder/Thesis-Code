using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie
using LinearAlgebra, Printf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

const DESING_KAPPA = 1e-4
const EPS_CUT = 1e-4

const NX = 64
const XMIN, XMAX = 0.0, 100.0
const T_END = 20.0
const OBS_TIMES = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0]
const CELL_INDICES = collect(1:NX)

const H1_CONST_ABOVE_INTERFACE = 0.75
const W0_LEFT_TRUE, W0_RIGHT_TRUE = 1.85, 0.10
const X_DAM = 50.0
const TRANSITION_WIDTH_W_TRUE = 10.0
const W0_INIT_CONST = 0.5

const B_AMP = 1.0
const B_CENTER = 50.0
const B_WIDTH = 10.0
const B_BACKGROUND = -1.0

const W_EPS = 0
const W_U1  = 1
const W_U2  = 0
const W_REG_H1 = 1e-4
const OBJ_SCALE = 1.0

const LBFGS_M = 10
const LBFGS_MAX_ITERS = 10
const LBFGS_G_SWITCH = 1-6

const GN_MAX_ITERS = 50
const GN_G_FINAL = 1e-7
const GN_DAMPING0 = 1e-8
const GN_ARMIJO_C1 = 1e-6
const GN_BACKTRACK = 0.8
const GN_MIN_STEP = 1e-2

const SAVE_DIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization"
mkpath(SAVE_DIR)

# ---------------------------------------------------------------------------
# Grid and profiles
# ---------------------------------------------------------------------------

function make_reference_grid()
    grid = SinFVM.CartesianGrid(NX; gc=2, boundary=SinFVM.WallBC(), extent=[XMIN XMAX])
    x = collect(SinFVM.cell_centers(grid))
    return x, x[2] - x[1]
end

const X_GRID, DX = make_reference_grid()

smooth_step_profile(x; left, right, center, width) =
    [right + (left - right) * (1 + tanh((center - xi) / width)) / 2 for xi in x]

smooth_bathymetry_profile(x; amp, center, width, background) =
    [background + amp * exp(-((xi - center) / width)^2) for xi in x]

const B_PROFILE = smooth_bathymetry_profile(
    X_GRID;
    amp=B_AMP,
    center=B_CENTER,
    width=B_WIDTH,
    background=B_BACKGROUND,
)

const W0_TRUE_PROFILE = smooth_step_profile(
    X_GRID;
    left=W0_LEFT_TRUE,
    right=W0_RIGHT_TRUE,
    center=X_DAM,
    width=TRANSITION_WIDTH_W_TRUE,
)

const EPS_TRUE_PROFILE = W0_TRUE_PROFILE .+ H1_CONST_ABOVE_INTERFACE
const LOWER_W0_PROFILE = B_PROFILE .+ EPS_CUT
const UPPER_W0_PROFILE = EPS_TRUE_PROFILE .- EPS_CUT

const W0_INIT_PROFILE = clamp.(
    fill(W0_INIT_CONST, NX),
    LOWER_W0_PROFILE,
    UPPER_W0_PROFILE,
)

project_w0(w0) = clamp.(w0, LOWER_W0_PROFILE, UPPER_W0_PROFILE)

# ---------------------------------------------------------------------------
# Bathymetry and simulator
# ---------------------------------------------------------------------------

function make_bathymetry(backend, grid, ::Type{T}) where {T}
    B_cells = T.(B_PROFILE)
    x_faces = SinFVM.cell_faces(grid; interior=false)
    B_face = similar(x_faces, T)

    gc = grid.ghostcells[1]
    B_pad = similar(B_face)

    B_pad[1:gc] .= B_cells[1]
    B_pad[gc+1:gc+NX] = B_cells
    B_pad[gc+NX+1:end] .= B_cells[end]

    for i in eachindex(B_face)
        B_face[i] = i < length(B_face) ? T(0.5) * (B_pad[i] + B_pad[i+1]) : B_pad[i]
    end

    return SinFVM.BottomTopography1D(B_face, backend, grid)
end

bathymetry_values(::Any, ::Type{T}) where {T} = T.(B_PROFILE)

function setup_twolayer_simulator(; backend=SinFVM.make_cpu_backend(), ε_profile, w0_profile)
    TT = promote_type(eltype(ε_profile), eltype(w0_profile))

    grid = SinFVM.CartesianGrid(NX; gc=2, boundary=SinFVM.WallBC(), extent=[XMIN XMAX])
    B = make_bathymetry(backend, grid, TT)

    eq = SinFVM.TwoLayerShallowWaterEquations1D(
        B;
        ρ1=TT(0.98),
        ρ2=TT(1.00),
        g=TT(9.81),
        depth_cutoff=TT(EPS_CUT),
        desingularizing_kappa=TT(DESING_KAPPA),
    )

    rec = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1))
    flux = SinFVM.PathConservativeCentralUpwind(eq)

    cs = SinFVM.ConservedSystem(
        backend,
        rec,
        flux,
        eq,
        grid,
        [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()],
    )

    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=0.4)

    εv = TT.(ε_profile)
    wv = TT.(w0_profile)
    Bv = TT.(B_PROFILE)

    initial = map(1:NX) do i
        w = max(wv[i], Bv[i] + TT(EPS_CUT))
        h1 = max(εv[i] - w, TT(EPS_CUT))
        @SVector([h1, zero(TT), w, zero(TT)])
    end

    SinFVM.set_current_state!(sim, initial)
    return sim
end

smooth_positive(h, κ) = 0.5 * (h + sqrt(h^2 + κ^2))
smooth_velocity(h, q, κ) = q / smooth_positive(h, κ)

function observable_fields(sim)
    st = SinFVM.current_interior_state(sim)
    T = eltype(st.h1)
    B = bathymetry_values(sim, T)
    κ = T(DESING_KAPPA)

    h1, q1, w, q2 = st.h1, st.q1, st.w, st.q2
    h2 = w .- B
    ε = h1 .+ w

    return (;
        ε,
        u1 = smooth_velocity.(h1, q1, κ),
        u2 = smooth_velocity.(h2, q2, κ),
        w,
        Bvals = B,
    )
end

reconstruct_from_observables(ε, u1, u2, w, B) = (;
    h1 = ε .- w,
    h2 = w .- B,
    q1 = (ε .- w) .* u1,
    q2 = (w .- B) .* u2,
)

# ---------------------------------------------------------------------------
# Observation machinery
# ---------------------------------------------------------------------------

@with_kw mutable struct ObservableRecorder{VT,IT,OT}
    obs_times::VT
    cell_indices::IT
    next_obs::Int = 1
    data::Vector{OT} = OT[]
end

function (cb::ObservableRecorder)(time, sim)
    t = ForwardDiff.value(time)

    while cb.next_obs <= length(cb.obs_times) && t + 1e-12 >= cb.obs_times[cb.next_obs]
        obs = observable_fields(sim)

        for i in cb.cell_indices
            push!(cb.data, obs.ε[i])
            push!(cb.data, obs.u1[i])
            push!(cb.data, obs.u2[i])
        end

        cb.next_obs += 1
    end
end

function simulate_observations(; t_end, w0_profile, ε_profile, obs_times, cell_indices)
    ADType = promote_type(eltype(w0_profile), eltype(ε_profile))

    sim = setup_twolayer_simulator(
        backend=SinFVM.make_cpu_backend(ADType),
        ε_profile=ADType.(ε_profile),
        w0_profile=ADType.(w0_profile),
    )

    recorder = ObservableRecorder(
        obs_times=obs_times,
        cell_indices=cell_indices,
        data=ADType[],
    )

    SinFVM.simulate_to_time(sim, t_end; callback=recorder)
    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"

    return recorder.data
end

# ---------------------------------------------------------------------------
# Synthetic observations and residual
# ---------------------------------------------------------------------------

const EXACT_OBS = simulate_observations(
    t_end=T_END,
    w0_profile=W0_TRUE_PROFILE,
    ε_profile=EPS_TRUE_PROFILE,
    obs_times=OBS_TIMES,
    cell_indices=CELL_INDICES,
)

const N_TRIPLES = length(EXACT_OBS) ÷ 3

const EPS_SCALE = max(maximum(abs.(EXACT_OBS[1:3:end])), 1e-2)
const U1_SCALE  = max(maximum(abs.(EXACT_OBS[2:3:end])), 1e-2)
const U2_SCALE  = max(maximum(abs.(EXACT_OBS[3:3:end])), 1e-2)

const MISFIT_SCALE_EPS = sqrt(W_EPS / N_TRIPLES) / EPS_SCALE
const MISFIT_SCALE_U1  = sqrt(W_U1  / N_TRIPLES) / U1_SCALE
const MISFIT_SCALE_U2  = sqrt(W_U2  / N_TRIPLES) / U2_SCALE
const REG_SCALE = sqrt(W_REG_H1 / DX)

println("Generated synthetic exact observations:")
println("  n_cells       = $NX")
println("  n_obs_times   = $(length(OBS_TIMES))")
println("  n_observables = $(length(EXACT_OBS))")
println("  bathymetry    = Gaussian bump")
println("Residual scaling:")
println("  EPS_SCALE = $EPS_SCALE")
println("  U1_SCALE  = $U1_SCALE")
println("  U2_SCALE  = $U2_SCALE")
println("  W_EPS     = $W_EPS")
println("  W_U1      = $W_U1")
println("  W_U2      = $W_U2")
println("  W_REG_H1  = $W_REG_H1")

function residual_vector(w0_vec)
    w0_profile = project_w0(w0_vec)

    pred = simulate_observations(
        t_end=T_END,
        w0_profile=w0_profile,
        ε_profile=EPS_TRUE_PROFILE,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    T = eltype(pred)
    nmis = length(pred)
    nreg = length(w0_profile) - 1
    r = Vector{T}(undef, nmis + nreg)

    @inbounds for k in 1:3:nmis
        r[k]   = T(MISFIT_SCALE_EPS) * (pred[k]   - EXACT_OBS[k])
        r[k+1] = T(MISFIT_SCALE_U1)  * (pred[k+1] - EXACT_OBS[k+1])
        r[k+2] = T(MISFIT_SCALE_U2)  * (pred[k+2] - EXACT_OBS[k+2])
    end

    @inbounds for i in 1:nreg
        r[nmis+i] = T(REG_SCALE) * (w0_profile[i+1] - w0_profile[i])
    end

    return r
end

function cost_function(w0_vec)
    r = residual_vector(w0_vec)
    J = OBJ_SCALE * 0.5 * dot(r, r)

    if J isa ForwardDiff.Dual
        @assert all(.!isnan.(J.partials)) "NaN in gradient"
    end

    return J
end

grad!(storage, w0_vec) = ForwardDiff.gradient!(storage, cost_function, w0_vec)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@with_kw mutable struct OptimizationHistory
    iter::Vector{Int} = Int[]
    phase::Vector{String} = String[]
    J::Vector{Float64} = Float64[]
    gnorm::Vector{Float64} = Float64[]
    w0_profiles::Vector{Vector{Float64}} = Vector{Float64}[]
end

function push_history!(history, iter, phase, w0_vec, J, gnorm)
    push!(history.iter, iter)
    push!(history.phase, phase)
    push!(history.J, Float64(J))
    push!(history.gnorm, Float64(gnorm))
    push!(history.w0_profiles, collect(Float64.(project_w0(w0_vec))))
    return history
end

optim_iteration(state) =
    hasproperty(state, :iteration) ? Int(getproperty(state, :iteration)) :
    hasproperty(state, :pseudo_iteration) ? Int(getproperty(state, :pseudo_iteration)) :
    0

optim_state_x(state) =
    hasproperty(state, :x) ? getproperty(state, :x) :
    error("Could not find parameter vector in Optim state.")

optim_state_value(state, w0) =
    hasproperty(state, :value) ? Float64(getproperty(state, :value)) :
    hasproperty(state, :f_x) ? Float64(getproperty(state, :f_x)) :
    Float64(cost_function(w0))

function make_history_callback(history)
    last_iter = Ref(-1)

    return function cb(state)
        k = optim_iteration(state)
        k == last_iter[] && return false
        last_iter[] = k

        w0 = project_w0(copy(optim_state_x(state)))
        J = optim_state_value(state, w0)
        g = ForwardDiff.gradient(cost_function, w0)

        push_history!(history, k, "LBFGS-Fminbox", w0, J, norm(g))
        return false
    end
end

# ---------------------------------------------------------------------------
# Gauss--Newton phase
# ---------------------------------------------------------------------------

function gauss_newton_phase(
    w0_init;
    history,
    max_iters=GN_MAX_ITERS,
    damping0=GN_DAMPING0,
    c1=GN_ARMIJO_C1,
    backtrack=GN_BACKTRACK,
    min_step=GN_MIN_STEP,
    g_tol=GN_G_FINAL,
)
    w0 = project_w0(copy(w0_init))
    μ = damping0
    t0 = time()

    println("\nGauss--Newton phase")
    println("-------------------")
    println("Iter     Function value   Gradient norm   Step length   Avg step w0   Time")

    for k in 1:max_iters
        r = residual_vector(w0)
        J = 0.5 * dot(r, r)

        Jr = ForwardDiff.jacobian(residual_vector, w0)
        g = Jr' * r
        gnorm = norm(g)

        if gnorm < g_tol
            push_history!(history, history.iter[end] + 1, "GN", w0, J, gnorm)
            println("GN converged on gradient norm.")
            break
        end

        δ = -((Jr' * Jr + μ * I) \ g)

        α = 1.0
        accepted = false
        wtrial = w0
        Jtrial = J

        while α >= min_step
            wcand = project_w0(w0 .+ α .* δ)
            rcand = residual_vector(wcand)
            Jcand = 0.5 * dot(rcand, rcand)

            if Jcand <= J + c1 * α * dot(g, δ)
                wtrial = wcand
                Jtrial = Jcand
                accepted = true
                break
            end

            α *= backtrack
        end

        accepted || begin
            println("GN iter $k: line search failed, stopping.")
            break
        end

        step_avg = norm(wtrial .- w0) / sqrt(length(w0))
        w0 = wtrial

        push_history!(history, history.iter[end] + 1, "GN", w0, Jtrial, gnorm)

        @printf(
            "%5d   %14.6e   %13.6e   %11.6f   %11.6e   %.3f\n",
            k, Jtrial, gnorm, α, step_avg, time() - t0
        )

        μ = Jtrial < J ? max(0.5μ, 1e-8) : min(10μ, 1e4)
    end

    println("\nExiting Gauss--Newton phase")
    println("Total GN time = $(round(time() - t0; digits=3)) seconds")

    return project_w0(w0)
end

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

w0_start = project_w0(W0_INIT_PROFILE)
g0 = ForwardDiff.gradient(cost_function, w0_start)

history = OptimizationHistory()
push_history!(history, 0, "INIT", w0_start, cost_function(w0_start), norm(g0))

opts = Optim.Options(
    store_trace=true,
    show_trace=true,
    show_every=1,
    iterations=LBFGS_MAX_ITERS,
    outer_iterations=1,
    g_tol=LBFGS_G_SWITCH,
    f_abstol=0.0,
    x_abstol=0.0,
    allow_f_increases=true,
    successive_f_tol=5,
    callback=make_history_callback(history),
)

result_lbfgs = optimize(
    cost_function,
    grad!,
    LOWER_W0_PROFILE,
    UPPER_W0_PROFILE,
    w0_start,
    Fminbox(LBFGS(; m=LBFGS_M)),
    opts,
)

w0_lbfgs = project_w0(copy(Optim.minimizer(result_lbfgs)))
J_lbfgs = cost_function(w0_lbfgs)
g_lbfgs = ForwardDiff.gradient(cost_function, w0_lbfgs)

println("\n=== After LBFGS-Fminbox phase ===")
println("Cost after LBFGS          = $(round(J_lbfgs; digits=12))")
println("Gradient norm after LBFGS = $(round(norm(g_lbfgs); digits=12))")

w0_opt_profile = gauss_newton_phase(w0_lbfgs; history=history)

final_cost = cost_function(w0_opt_profile)
final_grad = ForwardDiff.gradient(cost_function, w0_opt_profile)
profile_error = norm(w0_opt_profile .- W0_TRUE_PROFILE) / sqrt(NX)

println("\n=== Optimization complete ===")
println("Final raw cost   = $(round(final_cost; digits=12))")
println("Final grad norm  = $(round(norm(final_grad); digits=12))")
println("Profile L2 error = $(round(profile_error; digits=12))")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

function snapshot(w0_profile; ε_profile=EPS_TRUE_PROFILE, label="", t_end=T_END)
    sim = setup_twolayer_simulator(
        backend=SinFVM.make_cpu_backend(),
        ε_profile=Float64.(ε_profile),
        w0_profile=Float64.(w0_profile),
    )

    x = collect(SinFVM.cell_centers(sim.grid))
    SinFVM.simulate_to_time(sim, t_end)

    obs = observable_fields(sim)
    ε, u1, u2, w, B = collect(obs.ε), collect(obs.u1), collect(obs.u2), collect(obs.w), collect(obs.Bvals)
    rec = reconstruct_from_observables(ε, u1, u2, w, B)

    return (;
        x,
        ε,
        u1,
        u2,
        w,
        B,
        h1=collect(rec.h1),
        h2=collect(rec.h2),
        q1=collect(rec.q1),
        q2=collect(rec.q2),
        w0_profile=collect(w0_profile),
        label,
        t_end,
    )
end

function add_elevation!(ax, sn; legend_position=:lt)
    lines!(ax, sn.x, sn.ε, linewidth=2, label="ε")
    lines!(ax, sn.x, sn.w, linewidth=2, linestyle=:dash, label="w")
    lines!(ax, sn.x, sn.B, linewidth=2, label="B")
    axislegend(ax; position=legend_position)
end

function add_velocity!(ax, sn; legend_position=:lt)
    lines!(ax, sn.x, sn.u1, linewidth=2, label="u1")
    lines!(ax, sn.x, sn.u2, linewidth=2, label="u2")
    axislegend(ax; position=legend_position)
end

snap_ic  = snapshot(W0_INIT_PROFILE; label="initial condition", t_end=0.0)
snap_syn = snapshot(W0_TRUE_PROFILE; label="synthetic", t_end=T_END)
snap_opt = snapshot(w0_opt_profile; label="optimized", t_end=T_END)

fig = Figure(size=(1500, 1300), fontsize=22)

ax_prof = Axis(fig[1, 1:2], title="Initial interface profiles with bathymetry", xlabel="x", ylabel="elevation")
lines!(ax_prof, X_GRID, W0_TRUE_PROFILE, linewidth=3, linestyle=:dash, label="true w₀")
lines!(ax_prof, X_GRID, W0_INIT_PROFILE, linewidth=2, linestyle=:dot, label="initial guess")
lines!(ax_prof, X_GRID, w0_opt_profile, linewidth=3, label="optimized w₀")
lines!(ax_prof, X_GRID, B_PROFILE, linewidth=2, label="bathymetry B")
lines!(ax_prof, X_GRID, LOWER_W0_PROFILE, linewidth=1, linestyle=:dashdot, label="lower bound")
lines!(ax_prof, X_GRID, UPPER_W0_PROFILE, linewidth=1, linestyle=:dashdot, label="upper bound")
axislegend(ax_prof, position=:rb)

for (row, sn) in enumerate([snap_ic, snap_syn, snap_opt])
    ax_eps = Axis(
        fig[row + 1, 1],
        title="$(sn.label): free surface, interface and bathymetry at t=$(sn.t_end)",
        xlabel="x",
        ylabel="elevation",
    )

    ax_u = Axis(
        fig[row + 1, 2],
        title="$(sn.label): velocities at t=$(sn.t_end)",
        xlabel="x",
        ylabel="velocity",
    )

    pos = sn.label == "initial condition" ? :rt : :lt
    add_elevation!(ax_eps, sn; legend_position=pos)
    add_velocity!(ax_u, sn; legend_position=pos)
end

display(fig)

fig_file = joinpath(SAVE_DIR, "optimization_profile_result_bathymetry.png")
save(fig_file, fig)
println("Saved figure to: $fig_file")

# ---------------------------------------------------------------------------
# Optional animation
# ---------------------------------------------------------------------------

function animate_iteration_updates(history; filename, framerate=2)
    snaps = [
        snapshot(wprof; label="$(ph) iter $(k)", t_end=T_END)
        for (k, ph, wprof) in zip(history.iter, history.phase, history.w0_profiles)
    ]

    prof_obs = Observable(snaps[1].w0_profile)
    ε_obs = Observable(snaps[1].ε)
    w_obs = Observable(snaps[1].w)
    B_obs = Observable(snaps[1].B)
    u1_obs = Observable(snaps[1].u1)
    u2_obs = Observable(snaps[1].u2)

    iter_obs = Observable(history.iter[1])
    phase_obs = Observable(history.phase[1])
    J_obs = Observable(history.J[1])

    fig = Figure(size=(1500, 900), fontsize=22)
    Label(fig[1, 1:2], @lift("Phase: $phase_obs | iteration $iter_obs | J=$(round($J_obs; digits=10))"), fontsize=24)

    ax_prof = Axis(fig[2, 1:2], title="Recovered interface profile", xlabel="x", ylabel="elevation")
    lines!(ax_prof, X_GRID, W0_TRUE_PROFILE, linewidth=3, linestyle=:dash, label="true")
    lines!(ax_prof, X_GRID, W0_INIT_PROFILE, linewidth=2, linestyle=:dot, label="initial")
    lines!(ax_prof, X_GRID, prof_obs, linewidth=3, label="current")
    lines!(ax_prof, X_GRID, B_PROFILE, linewidth=2, label="bathymetry B")
    axislegend(ax_prof, position=:rb)

    ax_eps = Axis(fig[3, 1], title="Elevation fields", xlabel="x", ylabel="elevation")
    lines!(ax_eps, X_GRID, ε_obs, linewidth=2, label="ε")
    lines!(ax_eps, X_GRID, w_obs, linewidth=2, linestyle=:dash, label="w")
    lines!(ax_eps, X_GRID, B_obs, linewidth=2, label="B")
    axislegend(ax_eps, position=:lt)

    ax_u = Axis(fig[3, 2], title="Velocity fields", xlabel="x", ylabel="velocity")
    lines!(ax_u, X_GRID, u1_obs, linewidth=2, label="u1")
    lines!(ax_u, X_GRID, u2_obs, linewidth=2, label="u2")
    axislegend(ax_u, position=:lt)

    record(fig, filename, eachindex(snaps); framerate=framerate) do i
        sn = snaps[i]

        prof_obs[] = sn.w0_profile
        ε_obs[] = sn.ε
        w_obs[] = sn.w
        B_obs[] = sn.B
        u1_obs[] = sn.u1
        u2_obs[] = sn.u2

        iter_obs[] = history.iter[i]
        phase_obs[] = history.phase[i]
        J_obs[] = history.J[i]
    end

    return filename
end

anim_file = animate_iteration_updates(
    history;
    filename=joinpath(SAVE_DIR, "optimization_profile_iterations_bathymetry.mp4"),
    framerate=2,
)

println("Saved animation to: $anim_file")