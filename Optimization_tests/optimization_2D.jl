using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie
using LinearAlgebra, Printf

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

const NX, NY, GC = 64, 64, 2
const XMIN, XMAX = 0.0, 100.0
const YMIN, YMAX = 0.0, 50.0

const CFL_2D = 0.2
const DEPTH_CUT = 1e-4
const T_END = 6.0
const OBS_TIMES = [2.0, 4.0, 6.0]

const W_MIN, W_MAX = 0.5, 3.0
const W0_TRUE_CONST = 1.85
const W0_INIT_CONST = 1.00

const ETA_BACKGROUND = 2.80
const ETA_BUMP_AMP = 0.75
const ETA_BUMP_CENTER_X = 50.0
const ETA_BUMP_CENTER_Y = 25.0
const ETA_BUMP_RADIUS = 9.0

const CORIOLIS_F = 1e-4

const CELL_INDICES = [
    (8, 8), (12, 24), (16, 40),
    (24, 16), (32, 32), (40, 48),
    (48, 16), (56, 32), (60, 48),
]

# Only upper-layer velocities in the cost
const W_U1 = 1.0
const W_V1 = 1.0
const W_REG = 1e-10

const LBFGS_M = 5
const LBFGS_MAX_ITERS = 40
const LBFGS_G_TOL = 1e-10

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

function residual_vector_wlevel(wlevel_vec)
    wlevel = only(wlevel_vec)

    pred = simulate_observations(
        T_end=T_END,
        wlevel=wlevel,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    T = eltype(pred)

    nmis = length(pred)
    r = Vector{T}(undef, nmis + 1)

    @inbounds for k in 1:2:nmis
        r[k]   = T(SCALE_U1) * (pred[k]   - EXACT_OBS[k])
        r[k+1] = T(SCALE_V1) * (pred[k+1] - EXACT_OBS[k+1])
    end

    r[nmis + 1] = T(sqrt(W_REG)) * wlevel

    return r
end

raw_cost(wlevel_vec) = begin
    r = residual_vector_wlevel(wlevel_vec)
    0.5 * dot(r, r)
end

function cost_function(zvec)
    wlevel = to_w(only(zvec))
    J = raw_cost([wlevel])

    if J isa ForwardDiff.Dual
        @assert all(.!isnan.(J.partials)) "NaN in gradient"
    end

    return J
end

grad!(storage, zvec) = ForwardDiff.gradient!(storage, cost_function, zvec)

# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------

z_start = [from_w(W0_INIT_CONST)]

opts = Optim.Options(
    store_trace=true,
    show_trace=true,
    show_every=1,
    iterations=LBFGS_MAX_ITERS,
    g_tol=LBFGS_G_TOL,
    f_abstol=0.0,
    x_abstol=0.0,
    allow_f_increases=true,
)

result = optimize(
    cost_function,
    grad!,
    z_start,
    LBFGS(; m=LBFGS_M),
    opts,
)

z_opt = Optim.minimizer(result)
wlevel_opt = to_w(only(z_opt))

final_cost = raw_cost([wlevel_opt])
final_grad = ForwardDiff.gradient(cost_function, z_opt)
w_error = abs(wlevel_opt - W0_TRUE_CONST)

println("\n=== Optimization complete ===")
println("True w0 level      = $(round(W0_TRUE_CONST; digits=8))")
println("Initial w0 level   = $(round(W0_INIT_CONST; digits=8))")
println("Recovered w0 level = $(round(wlevel_opt; digits=8))")
println("Absolute error     = $(round(w_error; digits=12))")
println("Final raw cost     = $(round(final_cost; digits=12))")
println("Final grad norm    = $(round(norm(final_grad); digits=12))")

# -----------------------------------------------------------------------------
# Snapshot helper
# -----------------------------------------------------------------------------

function initial_eta_field(grid)
    xy_int = SinFVM.cell_centers(grid; interior=true)
    nx_int, ny_int = SinFVM.interior_size(grid)

    vals = [Float64(eta0_profile(xy_int[I], Float64)) for I in eachindex(xy_int)]

    return reshape(vals, nx_int, ny_int)
end

function initial_w_field(grid, wlevel)
    nx_int, ny_int = SinFVM.interior_size(grid)
    return fill(Float64(wlevel), nx_int, ny_int)
end

function snapshot(wlevel; label="")
    sim, eq, grid = setup_twolayer_simulator_2d(
        backend=SinFVM.make_cpu_backend(),
        wlevel=Float64(wlevel),
    )

    η0_init = initial_eta_field(grid)
    w0_init = initial_w_field(grid, wlevel)

    SinFVM.simulate_to_time(sim, T_END)

    obs = observable_fields(sim, eq, grid)

    return (;
        η0_init = Array(η0_init),
        w0_init = Array(w0_init),
        η = Array(obs.η),
        w = Array(obs.w),
        B = Array(obs.Bcell),
        u1 = Array(obs.u1),
        v1 = Array(obs.v1),
        u2 = Array(obs.u2),
        v2 = Array(obs.v2),
        label,
    )
end

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

snap_init = snapshot(W0_INIT_CONST; label="initial guess")
snap_true = snapshot(W0_TRUE_CONST; label="truth")
snap_opt  = snapshot(wlevel_opt; label="optimized")

fig = Figure(size=(1500, 1300), fontsize=18)

Label(
    fig[0, 1:3],
    "2-D constant-interface recovery using upper-layer velocities",
    fontsize=24,
)

snaps = [snap_init, snap_true, snap_opt]

for (row, sn) in enumerate(snaps)
    ax1 = Axis(fig[row, 1], title="$(sn.label): initial interface w₀")
    ax2 = Axis(fig[row, 2], title="$(sn.label): final free surface η")
    ax3 = Axis(fig[row, 3], title="$(sn.label): final upper-layer speed")

    speed1 = sqrt.(sn.u1.^2 .+ sn.v1.^2)

    heatmap!(ax1, sn.w0_init; colorrange=(W_MIN, W_MAX))
    heatmap!(ax2, sn.η; colorrange=(minimum(snap_true.η), maximum(snap_true.η)))
    heatmap!(ax3, speed1)

    hidedecorations!(ax1)
    hidedecorations!(ax2)
    hidedecorations!(ax3)
end

display(fig)

fig_file = joinpath(SAVE_DIR, "optimization_2d_constant_interface_u1v1.png")
save(fig_file, fig)

println("Saved figure to: $fig_file")