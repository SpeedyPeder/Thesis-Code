# ============================================================
# Example 5.4 (Barotropic Tidal Flow) — 1D Two-layer SWE
# Paper-faithful setup (domain [-10,10], flat bottom, periodic forcing on LEFT
# for h1 and h2, and q1/q2 by zero-order interpolation; open BC on RIGHT).
# ============================================================

using CairoMakie
using StaticArrays
using SinFVM

# ----------------------------
# Paper parameters (Example 5.4)
# ----------------------------
const UL_paper = (h1=0.69914, q1=-0.21977, h2=1.26932, q2=0.20656)
const UR_paper = (h1=0.37002, q1=-0.18684, h2=1.59310, q2=0.17416)

# Reference level Zref (paper eq. (5.5)) for flat bottom
compute_Zref(UL, UR) = -0.5 * (UL.h1 + UL.h2 + UR.h1 + UR.h2)

# Left boundary forcing (paper):
# h1(-10,t) = h1L + h1L*(0.03/|Zref|)*sin(π t / 50)
# h2(-10,t) = h2L + h1L*(0.03/|Zref|)*sin(π t / 50)
function left_boundary_heights(t, UL, Zref)
    amp = (0.03 / abs(Zref)) * sin(pi * t / 50.0)
    h1 = UL.h1 + UL.h1 * amp
    h2 = UL.h2 + UL.h1 * amp
    return h1, h2
end

# Initial Riemann data: UL for x<0, UR for x>=0
function riemann_ic(x, UL, UR)
    if x < 0.0
        return @SVector [UL.h1, UL.q1, UL.h2, UL.q2]
    else
        return @SVector [UR.h1, UR.q1, UR.h2, UR.q2]
    end
end

function bc_callback!(t, sim, grid, UL, UR)
    gc = SinFVM.ghost_cells(grid, SinFVM.XDIR)
    U  = SinFVM.current_state(sim)

    first_interior = gc + 1
    last_interior  = grid.totalcells[1] - gc

    Zref = compute_Zref(UL, UR)
    h1L, h2L = left_boundary_heights(t, UL, Zref) 

    q1L = U[first_interior][2]
    q2L = U[first_interior][4]

    Uleft = @SVector [h1L, q1L, h2L + Zref, q2L]

    @inbounds for i in 1:gc
        U[first_interior - i] = Uleft
    end

    Uright = U[last_interior]
    @inbounds for i in 1:gc
        U[last_interior + i] = Uright
    end

    return nothing
end

# ----------------------------
# Runner: return snapshots only
# ----------------------------
function run_example_5_4_snapshots(; nx=1000, gc=2, cfl=0.45, T=64.0,
    scheme=:pccu, ρ1=0.98, ρ2=1.0, g=9.81,
    checkpoints=[10.0, 25.0, 60.0, 64.0])

    backend = SinFVM.make_cpu_backend()
    grid = SinFVM.CartesianGrid(nx; gc=gc, extent=[-10.0 10.0], boundary=SinFVM.NeumannBC())
    x = SinFVM.cell_centers(grid; interior=true)

    Zref = compute_Zref(UL_paper, UR_paper)

    bottom = SinFVM.ConstantBottomTopography(Zref)
    equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=ρ1, ρ2=ρ2, g=g)

    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))

    numericalflux =
        scheme == :pccu ? SinFVM.PathConservativeCentralUpwind(equation) :
        scheme == :cu   ? SinFVM.CentralUpwind(equation) :
        error("scheme must be :cu or :pccu")

    bottom_src = SinFVM.SourceTermBottom()
    ncp_src    = SinFVM.SourceTermNonConservative()

    cs  = SinFVM.ConservedSystem(
        backend, reconstruction, numericalflux,
        equation, grid, [bottom_src, ncp_src]
    )

    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)

    function riemann_ic_w(xi)
        if xi < 0.0
            return @SVector [UL_paper.h1, UL_paper.q1, UL_paper.h2 + Zref, UL_paper.q2]
        else
            return @SVector [UR_paper.h1, UR_paper.q1, UR_paper.h2 + Zref, UR_paper.q2]
        end
    end

    initial = [riemann_ic_w(xi) for xi in x]
    SinFVM.set_current_state!(sim, initial)

    bc_callback!(0.0, sim, grid, UL_paper, UR_paper)

    function fields()
        st = SinFVM.current_interior_state(sim)

        h1 = collect(st.h1)
        w  = collect(st.w)

        ξ = h1 .+ w
        ω = w

        return (; ξ, ω)
    end

    snapshots = Dict{Float64, NamedTuple}()

    step_cb = (t, sim_) -> bc_callback!(t, sim_, grid, UL_paper, UR_paper)

    for ttarget in checkpoints
        SinFVM.simulate_to_time(sim, ttarget; callback=step_cb)
        bc_callback!(ttarget, sim, grid, UL_paper, UR_paper)

        snapshots[ttarget] = fields()

        println("$(uppercase(string(scheme))) reached checkpoint t=$ttarget")
    end

    return (; x, snapshots, sim)
end


# ----------------------------
# Plot CU and PCCU like paper
# ----------------------------
function plot_example_5_4_CU_vs_PCCU(; nx=1000, gc=2, cfl=0.45,
    checkpoints=[10.0, 25.0, 60.0, 64.0],
    ρ1=0.98, ρ2=1.0, g=9.81)

    cu = run_example_5_4_snapshots(
        nx=nx, gc=gc, cfl=cfl,
        scheme=:cu,
        checkpoints=checkpoints,
        ρ1=ρ1, ρ2=ρ2, g=g
    )

    pccu = run_example_5_4_snapshots(
        nx=nx, gc=gc, cfl=cfl,
        scheme=:pccu,
        checkpoints=checkpoints,
        ρ1=ρ1, ρ2=ρ2, g=g
    )

    x = cu.x

    fig = Figure(size=(900, 1050), fontsize=14)

    for (k, t) in enumerate(checkpoints)

        axϵ = Axis(
            fig[k, 1],
            title="t = $(Int(t))",
            xlabel="",
            ylabel=""
        )

        axw = Axis(
            fig[k, 2],
            title="t = $(Int(t))",
            xlabel="",
            ylabel=""
        )

        ϵ_cu   = cu.snapshots[t].ξ
        w_cu   = cu.snapshots[t].ω

        ϵ_pccu = pccu.snapshots[t].ξ
        w_pccu = pccu.snapshots[t].ω

        lines!(axϵ, x, ϵ_cu, linewidth=2, label="CU")
        lines!(axϵ, x, ϵ_pccu, linewidth=2, label="PCCU")

        lines!(axw, x, w_cu, linewidth=2, label="CU")
        lines!(axw, x, w_pccu, linewidth=2, label="PCCU")

        xlims!(axϵ, -10, 10)
        xlims!(axw, -10, 10)

        ylims!(axϵ, -0.03, 0.03)
        ylims!(axw, -0.9, -0.2)

        if k == length(checkpoints)
            axϵ.xlabel = "x"
            axw.xlabel = "x"
        end

        if k ≥ 3
            axislegend(axϵ, position=:rt)
        else
            axislegend(axϵ, position=:rb)
        end

        axislegend(axw, position=:rb)
    end

    Label(fig[1:length(checkpoints), 0],
        "Water surface ϵ",
        rotation=pi/2)

    Label(fig[1:length(checkpoints), 3],
        "Interface w",
        rotation=-pi/2)

    display(fig)

    return fig, cu, pccu
end


# ============================================================
# Run both CU and PCCU, and compare them in the same figure
# ============================================================
fig, cu, pccu = plot_example_5_4_CU_vs_PCCU(
    nx=1000,
    gc=2,
    cfl=0.45,
    ρ1=0.98,
    ρ2=1.0,
    g=9.81,
    checkpoints=[10.0, 25.0, 60.0, 64.0]
)

#Save the figure to folder
save(
    raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Plots- Two-Layer\CU_VS_PCCU_comparison.png",
    fig
)