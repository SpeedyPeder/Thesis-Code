# ============================================================
# Internal Dam-Break Baseline — 1D Two-layer SWE
# Flat free surface, large interface jump, Neumann boundaries.
# Saves snapshot PNG and animation MP4 to specified folder.
# ============================================================

using CairoMakie, StaticArrays, SinFVM

const OUTDIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Plots- Two-Layer"
const SNAPSHOT_PNG = "internal_dambreak_snapshot.png"
const ANIMATION_MP4 = "internal_dambreak_animation.mp4"
mkpath(OUTDIR)

const LEFT_STATE  = (h1=1.0,  q1=0.0, h2=39.0, q2=0.0)
const RIGHT_STATE = (h1=39.0, q1=0.0, h2=1.0,  q2=0.0)

function ic(x, x0, UL, UR, B)
    U = x < x0 ? UL : UR
    return @SVector [U.h1, U.q1, U.h2 + B, U.q2]
end

function run_internal_dambreak(; nx=1200, gc=2, cfl=0.45, T=2.0, scheme=:pccu,
    ρ1=0.90, ρ2=1.0, g=9.81, xmin=-10.0, xmax=10.0, x0=0.0, B=0.0,
    snapshot_times=[0.5, 1.0, 2.0], nframes=240, framerate=30, save_outputs=true)

    backend = SinFVM.make_cpu_backend()
    grid = SinFVM.CartesianGrid(nx; gc=gc, extent=[xmin xmax], boundary=SinFVM.NeumannBC())
    x = SinFVM.cell_centers(grid; interior=true)

    bottom = SinFVM.ConstantBottomTopography(B)
    eq = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=ρ1, ρ2=ρ2, g=g)
    rec = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))
    flux = scheme == :pccu ? SinFVM.PathConservativeCentralUpwind(eq) :
           scheme == :cu   ? SinFVM.CentralUpwind(eq) : error("scheme must be :pccu or :cu")
    cs = SinFVM.ConservedSystem(backend, rec, flux, eq, grid,
        [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()])

    make_sim() = begin
        sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)
        SinFVM.set_current_state!(sim, [ic(xi, x0, LEFT_STATE, RIGHT_STATE, B) for xi in x])
        sim
    end

    fields(s) = begin
        st = SinFVM.current_interior_state(s)
        h1, q1, w, q2 = collect(st.h1), collect(st.q1), collect(st.w), collect(st.q2)
        h2 = w .- B
        (; h1, h2, q1, q2, u1=q1 ./ h1, u2=q2 ./ h2, ϵ=h1 .+ w, ω=w)
    end

    # ---------------- Snapshot at three times ----------------
    sim = make_sim()
    fig = Figure(size=(1000, 1000), fontsize=22)
    Label(fig[0, 1], "Internal dam-break baseline: surface and interface", fontsize=28)
    ax = Axis(fig[1, 1], xlabel="x", ylabel="height [m]", aspect=nothing)
    xlims!(ax, xmin, xmax); ylims!(ax, B - 1.0, 50.0)

    tnow = 0.0
    for t in snapshot_times
        t > tnow && (SinFVM.simulate_to_time(sim, t); tnow = t)
        f = fields(sim); ts = string(round(t; digits=2))
        lines!(ax, x, f.ϵ, linewidth=3, label="surface ϵ, t=$ts")
        lines!(ax, x, f.ω, linewidth=3, linestyle=:dash, label="interface ω, t=$ts")
    end
    axislegend(ax, position=:lb, framevisible=true)

    if save_outputs
        save(joinpath(OUTDIR, SNAPSHOT_PNG), fig)
        println("Saved PNG: ", joinpath(OUTDIR, SNAPSHOT_PNG))
    end
    display(fig)

    # ---------------- Animation ----------------
    simA = make_sim(); f0 = fields(simA)
    figA = Figure(size=(1000, 1000), fontsize=20)
    Label(figA[0, 1], "Internal dam-break baseline", fontsize=26, tellwidth=false)
    axA = Axis(figA[1, 1], xlabel="x", ylabel="height [m]", aspect=nothing)
    xlims!(axA, xmin, xmax); ylims!(axA, B - 1.0, 50.0)

    ϵ = Observable(f0.ϵ); ω = Observable(f0.ω)
    lines!(axA, x, ϵ, linewidth=3, label="surface ϵ")
    lines!(axA, x, ω, linewidth=3, linestyle=:dash, label="interface ω")
    axislegend(axA, position=:lb, framevisible=true)

    tA = 0.0
    if save_outputs
        record(figA, joinpath(OUTDIR, ANIMATION_MP4), range(0, T; length=nframes); framerate=framerate) do t
            t > tA && (SinFVM.simulate_to_time(simA, t); tA = t)
            f = fields(simA); ϵ[] = f.ϵ; ω[] = f.ω
        end
        println("Saved animation: ", joinpath(OUTDIR, ANIMATION_MP4))
    end

    return (; fig, sim, grid, x, B)
end

result = run_internal_dambreak(nx=1200, gc=2, T=2.0, cfl=0.45, scheme=:pccu,
    ρ1=0.90, ρ2=1.0, g=9.81, B=0.0,
    snapshot_times=[0.0, 0.2, 0.4], nframes=240, framerate=30, save_outputs=true)
