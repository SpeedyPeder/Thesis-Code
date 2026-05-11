# ============================================================
# Constant-interface optimization — 1D two-layer SWE
# Optimizes only scalar w0 with bounds B < w0 < ε.
# ============================================================

using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie

const DESING_KAPPA = 1e-4
const EPS_CUT = 1e-4
const NX = 64
const XMIN, XMAX, X_DAM = 0.0, 100.0, 50.0
const T_END = 20.0
const H1_LEFT, H1_RIGHT = 1.0, 0.20
const W0_TRUE, W0_INIT = 0.10, 0.25
const OBS_TIMES = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
const CELL_INDICES = collect(1:NX)
const W_EPS, W_U1, W_U2 = 0.0, 1, 0.0
const SAVE_DIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization"
mkpath(SAVE_DIR)

const B0 = 0.0
const EPS_LEFT_TRUE  = H1_LEFT  + W0_TRUE
const EPS_RIGHT_TRUE = H1_RIGHT + W0_TRUE
const W0_MIN = B0 + EPS_CUT
const W0_MAX = min(EPS_LEFT_TRUE, EPS_RIGHT_TRUE) - EPS_CUT

function setup_sim(; backend=SinFVM.make_cpu_backend(), w0)
    T = typeof(w0)
    grid = SinFVM.CartesianGrid(NX; gc=2, boundary=SinFVM.WallBC(), extent=[XMIN XMAX])
    x = SinFVM.cell_centers(grid)
    B = SinFVM.ConstantBottomTopography(T(B0))

    eq = SinFVM.TwoLayerShallowWaterEquations1D(B; ρ1=T(0.98), ρ2=T(1.0), g=T(9.81),
        depth_cutoff=T(EPS_CUT), desingularizing_kappa=T(DESING_KAPPA))
    rec = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1))
    cs = SinFVM.ConservedSystem(backend, rec, SinFVM.PathConservativeCentralUpwind(eq), eq, grid,
        [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()])
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=0.1)

    w = clamp(T(w0), T(W0_MIN), T(W0_MAX))

    initial = map(x) do xi
        ε0 = xi < X_DAM ? T(EPS_LEFT_TRUE) : T(EPS_RIGHT_TRUE)
        h1 = max(ε0 - w, T(EPS_CUT))
        h2 = max(w - T(B0), T(EPS_CUT))
        @SVector [h1, zero(T), h2 + T(B0), zero(T)]
    end

    SinFVM.set_current_state!(sim, initial)
    return sim
end

#Desingularization
smooth_positive(h, κ) = 0.5 * (h + sqrt(h^2 + κ^2))
smooth_velocity(h, q, κ) = q / smooth_positive(h, κ)

function obs_fields(sim)
    st = SinFVM.current_interior_state(sim)
    T = eltype(st.h1); κ = T(DESING_KAPPA); B = fill(T(B0), length(st.h1))
    h1, q1, w, q2 = st.h1, st.q1, st.w, st.q2
    h2 = w .- B
    return (; ε=h1 .+ w, w, B, u1=smooth_velocity.(h1, q1, κ), u2=smooth_velocity.(h2, q2, κ))
end

@with_kw mutable struct Recorder{VT,IT,OT}
    obs_times::VT; cell_indices::IT; next_obs::Int = 1; data::Vector{OT} = OT[]
end

function (r::Recorder)(time, sim)
    t = ForwardDiff.value(time)
    while r.next_obs <= length(r.obs_times) && t + 1e-12 >= r.obs_times[r.next_obs]
        o = obs_fields(sim)
        for i in r.cell_indices
            push!(r.data, o.ε[i]); push!(r.data, o.u1[i]); push!(r.data, o.u2[i])
        end
        r.next_obs += 1
    end
end

function simulate_obs(w0)
    T = typeof(w0)
    sim = setup_sim(backend=SinFVM.make_cpu_backend(T), w0=T(w0))
    rec = Recorder(obs_times=OBS_TIMES, cell_indices=CELL_INDICES, data=T[])
    SinFVM.simulate_to_time(sim, T(T_END); callback=rec)
    @assert rec.next_obs == length(OBS_TIMES) + 1 "Not all observation times were recorded"
    return rec.data
end

const EXACT_OBS = simulate_obs(W0_TRUE)
const NTRIP = length(EXACT_OBS) ÷ 3
const S_EPS, S_U1, S_U2 = sqrt(W_EPS / NTRIP), sqrt(W_U1 / NTRIP), sqrt(W_U2 / NTRIP)

function residual_vector(wvec)
    pred = simulate_obs(wvec[1])
    T = eltype(pred); r = similar(pred)
    @inbounds for k in 1:3:length(pred)
        r[k]   = T(S_EPS) * (pred[k]   - EXACT_OBS[k])
        r[k+1] = T(S_U1)  * (pred[k+1] - EXACT_OBS[k+1])
        r[k+2] = T(S_U2)  * (pred[k+2] - EXACT_OBS[k+2])
    end
    return r
end

cost(wvec) = 0.5 * sum(abs2, residual_vector(wvec))
grad!(g, wvec) = ForwardDiff.gradient!(g, cost, wvec)

@with_kw mutable struct History
    iter::Vector{Int}=Int[]; w0::Vector{Float64}=Float64[]; J::Vector{Float64}=Float64[]
end

optim_iter(s) = hasproperty(s,:iteration) ? Int(s.iteration) : hasproperty(s,:pseudo_iteration) ? Int(s.pseudo_iteration) : 0
optim_x(s) = hasproperty(s,:x) ? s.x : Optim.minimizer(s)
optim_val(s, x) = hasproperty(s,:value) ? Float64(s.value) : hasproperty(s,:f_x) ? Float64(s.f_x) : Float64(cost(x))


function history_callback(hist)
    first_callback = Ref(true)

    return s -> begin
        x = optim_x(s)

        # Skip duplicate initial callback from Optim/Fminbox
        if first_callback[]
            first_callback[] = false
            if isapprox(Float64(x[1]), hist.w0[end]; atol=1e-14)
                return false
            end
        end

        # Continue from last stored iteration
        k = hist.iter[end] + 1

        push!(hist.iter, k)
        push!(hist.w0, Float64(x[1]))
        push!(hist.J, optim_val(s, x))
        return false
    end
end

hist = History()
push!(hist.iter, 0); push!(hist.w0, W0_INIT); push!(hist.J, Float64(cost([W0_INIT])))

result = optimize(cost, grad!, [W0_MIN], [W0_MAX], [W0_INIT],
    Fminbox(LBFGS(; m=5)),
    Optim.Options(show_trace=true, show_every=1, iterations=30, g_tol=1e-8,
        allow_f_increases=false, callback=history_callback(hist)))

w_opt = Optim.minimizer(result)[1]
println("\n=== Optimization complete ===")
println("Bounds       = [$W0_MIN, $W0_MAX]")
println("True w0      = ", W0_TRUE)
println("Recovered w0 = ", w_opt)
println("Abs. error   = ", abs(w_opt - W0_TRUE))
println("Final cost   = ", cost([w_opt]))

function snapshot(w0; label="", t=T_END)
    sim = setup_sim(w0=Float64(w0)); x = collect(SinFVM.cell_centers(sim.grid))
    SinFVM.simulate_to_time(sim, t)
    o = obs_fields(sim)
    return (; x, ε=collect(o.ε), w=collect(o.w), B=collect(o.B), u1=collect(o.u1), u2=collect(o.u2), label, t)
end

add_elev!(ax, s) = (lines!(ax,s.x,s.ε,label="ε"); lines!(ax,s.x,s.w,linestyle=:dash,label="w"); lines!(ax,s.x,s.B,label="B"); axislegend(ax,position=:lb))
add_vel!(ax, s) = (lines!(ax,s.x,s.u1,label="u₁"); lines!(ax,s.x,s.u2,label="u₂"); axislegend(ax,position=:lt))

snaps = [snapshot(W0_INIT; label="initial", t=0.0), snapshot(W0_TRUE; label="synthetic", t=T_END), snapshot(w_opt; label="optimized", t=T_END)]

fig = Figure(size=(1500,1100), fontsize=20)
ax0 = Axis(fig[1,1:2], title="Convergence of constant interface parameter w₀", xlabel="iteration", ylabel="w₀")
lines!(ax0, hist.iter, hist.w0, label="recovered w₀"); scatter!(ax0, hist.iter, hist.w0)
hlines!(ax0, [W0_TRUE], linestyle=:dash, label="true w₀"); axislegend(ax0, position=:rt)

for (j,s) in enumerate(snaps)
    axε = Axis(fig[j+1,1], title="$(s.label): ε, w and B at t=$(s.t)", xlabel="x", ylabel="elevation")
    axu = Axis(fig[j+1,2], title="$(s.label): velocities at t=$(s.t)", xlabel="x", ylabel="velocity")
    add_elev!(axε, s); add_vel!(axu, s)
end

display(fig)
save(joinpath(SAVE_DIR, "constant_w0_optimization.png"), fig)



function animate_history(hist; filename=joinpath(SAVE_DIR,"constant_w0_optimization.mp4"), framerate=2)
    ss = [snapshot(w; label="iter $k", t=T_END) for (k,w) in zip(hist.iter,hist.w0)]
    ε,w,B,u1,u2 = Observable(ss[1].ε), Observable(ss[1].w), Observable(ss[1].B), Observable(ss[1].u1), Observable(ss[1].u2)
    it,wcur,Jcur = Observable(hist.iter[1]), Observable(hist.w0[1]), Observable(hist.J[1])
    fig = Figure(size=(1500,900), fontsize=20)
    Label(fig[0,1:2], @lift("iter $it | w₀=$(round($wcur;digits=8)) | J=$(round($Jcur;digits=10))"), fontsize=24)
    ax1 = Axis(fig[1,1], title="Elevation fields", xlabel="x", ylabel="elevation")
    ax2 = Axis(fig[1,2], title="Velocities", xlabel="x", ylabel="velocity")
    lines!(ax1, ss[1].x, ε, label="ε"); lines!(ax1, ss[1].x, w, linestyle=:dash, label="w"); lines!(ax1, ss[1].x, B, label="B"); axislegend(ax1, position=:lt)
    lines!(ax2, ss[1].x, u1, label="u₁"); lines!(ax2, ss[1].x, u2, label="u₂"); axislegend(ax2, position=:lt)
    record(fig, filename, eachindex(ss); framerate=framerate) do i
        s = ss[i]; ε[]=s.ε; w[]=s.w; B[]=s.B; u1[]=s.u1; u2[]=s.u2
        it[]=hist.iter[i]; wcur[]=hist.w0[i]; Jcur[]=hist.J[i]
    end
    return filename
end

anim = animate_history(hist)
println("Saved figure: ", joinpath(SAVE_DIR, "constant_w0_optimization.png"))
println("Saved animation: ", anim)