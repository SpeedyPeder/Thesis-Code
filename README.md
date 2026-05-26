# Master's Thesis Code

This repository contains the numerical code used in the master's thesis

**Finite-Volume Methods and Calibration of Two-Layer Shallow-Water Models**

by **Peder Aas Vårheim**.

The code contains implementations and test cases for finite-volume simulations of shallow-water models, with particular focus on the two-layer shallow-water equations and optimization-based calibration of the internal interface. The numerical examples are mainly written in Julia and use [`SinFVM.jl`](https://github.com/sintefmath/SinFVM.jl) together with optimization and automatic differentiation tools.

The repository contains both scripts used directly to generate figures and tables in the thesis, and additional test cases used during development.

## Repository structure

The repository is organized into three main folders.

### `Additional_tests`

This folder contains additional numerical tests that are not directly used to produce figures in the thesis. These scripts are included for completeness and were used to test, compare, and verify different parts of the implementation during the project.

The files in this folder are not required to reproduce the main thesis results, but they may be useful for understanding intermediate experiments and implementation choices.

### `CU_PCCU_and_hyperbolicity_tests`

This folder contains scripts related to the numerical treatment of the two-layer shallow-water equations. In particular, it includes tests comparing the standard central-upwind method and the path-conservative central-upwind method, as well as experiments related to loss of hyperbolicity.

Important files include:

- `1DSWE_PCCU_CU.jl`  
  One-dimensional comparison of the central-upwind and path-conservative central-upwind methods.

- `Internal_Dambreak_baseline.jl`  
  Script used to generate the one-dimensional internal dam-break baseline example.

- `shallow_water_twolayer_1D_hyperbolic.jl`  
  Script related to hyperbolicity tests for the one-dimensional two-layer shallow-water system.

These scripts are mainly connected to the numerical-methodology part of the thesis.

### `Optimization_tests`

This folder contains the scripts used to generate the optimization results presented in the thesis. These are the most important files for reproducing the numerical experiments.

Important files include:

- `optimization_cont_1D.jl`  
  One-dimensional optimization of a continuous interface profile.

- `optimization_cont_1D_bathymetry.jl`  
  One-dimensional optimization of a continuous interface profile with bathymetry.

- `optimization_sparse_1D.jl`  
  One-dimensional optimization using a sparse representation of the interface.

- `optimization_1D.jl`  
  General one-dimensional optimization test script.

- `2D-opt_constant.jl`  
  Two-dimensional optimization with a constant interface parameter.

- `2D-opt_continous.jl`  
  Two-dimensional optimization with a continuous interface profile.

- `2D-opt_internal.jl`  
  Two-dimensional interface-driven motion test.

- `2D-opt_initialguess_pc.jl`  
  Two-dimensional initial-guess sensitivity experiment.

- `2D-opt_predictive_skill_pc.jl`  
  Two-dimensional predictive-skill experiment, where the optimized interface is tested outside the observation window.

The suffix `_pc` indicates scripts adapted for local PC runs, as they were intended to run on the IDUN computing cluster.

## Requirements

The code is written in Julia. The main dependencies are listed in `Project.toml` and `Manifest.toml`.

The most important Julia packages used are:

- `SinFVM.jl`
- `StaticArrays.jl`
- `ForwardDiff.jl`
- `Optim.jl`
- `Parameters.jl`
- `LinearAlgebra`
- `CairoMakie.jl`

To activate the project environment, run Julia from the repository root and use:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
