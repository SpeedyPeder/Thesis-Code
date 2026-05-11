using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie
using LinearAlgebra, Printf

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

const NX, NY, GC = 32, 32, 2
const XMIN, XMAX = 0.0, 100.0
const YMIN, YMAX = 0.0, 50.0

const CFL_2D = 0.2
const DEPTH_CUT = 1e-4
const T_END = 6.0
const OBS_TIMES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

const W_MIN, W_MAX = 0.5, 3.0
const W0_TRUE_CONST = 1.85
const W0_INIT_CONST = 1.00

const ETA_BACKGROUND = 2.80
const ETA_BUMP_AMP = 0.75
const ETA_BUMP_CENTER_X = 50.0
const ETA_BUMP_CENTER_Y = 25.0
const ETA_BUMP_RADIUS = 9.0

const CORIOLIS_F = 1e-4

const CELL_INDICES = [(8, 8), (16, 16), (24, 24)]

# Only upper-layer velocities in the cost (less weight on velocity variations)
const W_U1 = 1.0
const W_V1 = 0.0
const W_REG = 1e-10

const LBFGS_M = 5
const LBFGS_MAX_ITERS = 20
const LBFGS_G_TOL = 1e-8

const SAVE_DIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization"
mkpath(SAVE_DIR)

# -----------------------------------------------------------------------------
# Bounded scalar transform
# -----------------------------------------------------------------------------

σ(z) = inv(one(z) + exp(-z))
to_w(z) = W_MIN + (W_MAX - W_MIN) * σ(z)
from_w(w) = log((w - W_MIN) / (W_MAX - w))

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
# Initial free surface
# -----------------------------------------------------------------------------

function eta0_profile(xy, ::Type{T}) where {T}
    x, y = xy

    dx = (T(x) - T(ETA_BUMP_CENTER_X)) / T(ETA_BUMP_RADIUS)
    dy = (T(y) - T(ETA_BUMP_CENTER_Y)) / T(ETA_BUMP_RADIUS)

    return T(ETA_BACKGROUND) + T(ETA_BUMP_AMP) * exp(-(dx^2 + dy^2))
end

# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------

function setup_twolayer_simulator_2d(; backend=SinFVM.make_cpu_backend(), wlevel)
    T = typeof(wlevel)

    grid = SinFVM.CartesianGrid(
        NX,
        NY;
        gc=GC,
        boundary=SinFVM.PeriodicBC(),
        extent=[XMIN XMAX; YMIN YMAX],
    )

    bottom = make_bottom_cos_sin_2d(; backend=backend, grid=grid)

    eq = SinFVM.TwoLayerShallowWaterEquations2D(
        bottom;
        ρ1=T(1.00),
        ρ2=T(1.02),
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

    εh = T(DEPTH_CUT)

    initial = [begin
        η0 = eta0_profile(xy_int[I], T)

        lower_w = T(B_int[I]) + εh
        upper_w = η0 - εh
        w0 = clamp(T(wlevel), lower_w, upper_w)

        h1 = η0 - w0

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
    η = h1 .+ w

    u1 = SinFVM.desingularize.(Ref(eq), h1, q1)
    v1 = SinFVM.desingularize.(Ref(eq), h1, p1)
    u2 = SinFVM.desingularize.(Ref(eq), h2, q2)
    v2 = SinFVM.desingularize.(Ref(eq), h2, p2)

    return (; Bcell, η, w, h1, h2, u1, v1, u2, v2)
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
println("  true w0       = $W0_TRUE_CONST")
println("  initial w0    = $W0_INIT_CONST")
println("  n_observables = $(length(EXACT_OBS))")
println("  observables   = u1, v1 only")
println("  boundary      = periodic")

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
        r[k]   = T(SCALE_U1) * (pred[k]   - EXACT_OBS[k])
        r[k+1] = T(SCALE_V1) * (pred[k+1] - EXACT_OBS[k+1])
    end

    return r
end

raw_cost(wlevel) = 0.5 * sum(abs2, residual_vector_wlevel(wlevel))
cost(wvec) = raw_cost(wvec[1])
grad!(g, wvec) = ForwardDiff.gradient!(g, cost, wvec)

# History tracking
@with_kw mutable struct History
    iter::Vector{Int}=Int[]; w0::Vector{Float64}=Float64[]; J::Vector{Float64}=Float64[]
end

optim_iter(s) = hasproperty(s,:iteration) ? Int(s.iteration) : 0
optim_x(s) = hasproperty(s,:x) ? s.x : Optim.minimizer(s)
optim_val(s, x) = hasproperty(s,:value) ? Float64(s.value) : Float64(cost(x))

function history_callback(hist)
    first_callback = Ref(true)
    return s -> begin
        x = optim_x(s)
        if first_callback[]
            first_callback[] = false
            if isapprox(Float64(x[1]), hist.w0[end]; atol=1e-14)
                return false
            end
        end
        k = hist.iter[end] + 1
        push!(hist.iter, k)
        push!(hist.w0, Float64(x[1]))
        push!(hist.J, optim_val(s, x))
        return false
    end
end

# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------

hist = History()
push!(hist.iter, 0); push!(hist.w0, W0_INIT_CONST); push!(hist.J, Float64(cost([W0_INIT_CONST])))

result = optimize(
    cost,
    grad!,
    [W_MIN],
    [W_MAX],
    [W0_INIT_CONST],
    Fminbox(LBFGS(; m=LBFGS_M)),
    Optim.Options(
        show_trace=true,
        show_every=1,
        iterations=LBFGS_MAX_ITERS,
        g_tol=LBFGS_G_TOL,
        allow_f_increases=false,
        callback=history_callback(hist)
    ),
)

w_opt = Optim.minimizer(result)[1]

println("\n=== Optimization complete ===")
println("Bounds       = [$W_MIN, $W_MAX]")
println("True w0      = $W0_TRUE_CONST")
println("Recovered w0 = $(round(w_opt; digits=8))")
println("Abs. error   = $(round(abs(w_opt - W0_TRUE_CONST); digits=12))")
println("Final cost   = $(round(cost([w_opt]); digits=12))")

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
        η=Array(obs.η),
        u1=Array(obs.u1),
        v1=Array(obs.v1),
        label,
        t,
    )
end

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

function snapshot(wlevel; label="", t=T_END)
    sim, eq, grid = setup_twolayer_simulator_2d(
        backend=SinFVM.make_cpu_backend(),
        wlevel=Float64(wlevel),
    )
    SinFVM.simulate_to_time(sim, t)
    obs = observable_fields(sim, eq, grid)
    return (;
        η=Array(obs.η),
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

fig = Figure(size=(1500, 1100), fontsize=18)

# Convergence plot
ax_conv = Axis(fig[1, 1:2], title="Convergence of interface parameter w₀", xlabel="iteration", ylabel="w₀")
lines!(ax_conv, hist.iter, hist.w0, label="recovered w₀")
scatter!(ax_conv, hist.iter, hist.w0)
hlines!(ax_conv, [W0_TRUE_CONST], linestyle=:dash, label="true w₀")
axislegend(ax_conv, position=:rt)

# Snapshots
for (j, s) in enumerate(snaps)
    ax_η = Axis(fig[j+1, 1], title="$(s.label): free surface η at t=$(s.t)")
    ax_speed = Axis(fig[j+1, 2], title="$(s.label): upper-layer speed at t=$(s.t)")
    
    speed_u1 = sqrt.(s.u1.^2 .+ s.v1.^2)
    
    heatmap!(ax_η, s.η)
    heatmap!(ax_speed, speed_u1)
    
    hidedecorations!(ax_η)
    hidedecorations!(ax_speed)
end

display(fig)

fig_file = joinpath(SAVE_DIR, "optimization_2d_constant_interface.png")
save(fig_file, fig)

println("Saved figure to: $fig_file")