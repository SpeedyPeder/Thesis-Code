# Master's Thesis Code

This repository contains the numerical code used in the master's thesis

**Finite-Volume Methods and Calibration of Two-Layer Shallow-Water Models**

by **Peder Aas Vårheim**.

The code contains implementations and test cases for finite-volume simulations of shallow-water models, with particular focus on the two-layer shallow-water equations and optimization-based calibration of the internal interface. The numerical examples are mainly written in Julia and use `VolumeFluxes.jl` together with optimization and automatic differentiation tools.

The repository contains both scripts used directly to generate figures and tables in the thesis, and additional test cases used during development. Some scripts were used for exploratory testing and are included for completeness, but were not part of the final numerical evaluation.

## Repository structure

The repository is organized into four main folders.

### `Optimization_tests`

This folder contains the scripts used to generate the main optimization results presented in the thesis. These are the most important files for reproducing the numerical experiments.

Important files include:

- `optimization_1D.jl`  
  One-dimensional optimization test script.

- `optimization_cont_1D.jl`  
  One-dimensional optimization of a continuous interface profile.

- `optimization_cont_1D_bathymetry.jl`  
  One-dimensional optimization of a continuous interface profile with bathymetry.

- `optimization_sparse_1D.jl`  
  One-dimensional optimization using a sparse representation of the interface.

- `2D_opt_constant.jl`  
  Two-dimensional calibration with a constant interface parameter.

- `2D-opt_continous.jl`  
  Two-dimensional calibration with a continuous interface profile.

- `2D_initialguess_error_plots.jl`  
  Script for producing combined error plots for the initial-guess sensitivity experiment.

- `2D_predictive_skill_uniform_current.jl`  
  Two-dimensional predictive-skill experiment with a uniform background current, where the calibrated interface is tested outside the observation window.

These scripts are connected to the main calibration results discussed in the thesis.

### `Additional_2D_optimization`

This folder contains additional two-dimensional optimization scripts developed during the project. These scripts were used for exploratory tests of alternative two-dimensional setups, but were not included in the final numerical evaluation in the thesis.

Important files include:

- `2D-opt_initialguess_pc.jl`  
  Earlier two-dimensional initial-guess sensitivity script adapted for local PC runs.

- `2D-opt_internal.jl`  
  Two-dimensional interface-driven motion test.

- `2D-opt_predictive_skill_pc.jl`  
  Earlier two-dimensional predictive-skill script adapted for local PC runs.

- `2D_predictive_skill_periodic_errors.jl`  
  Script for periodic predictive-skill error analysis.

These scripts may be useful for understanding the development of the final two-dimensional experiments, but they are not required to reproduce the main thesis results.

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

### `Additional_tests`

This folder contains additional numerical tests that are not directly used to produce figures in the thesis. These scripts are included for completeness and were used to test, compare, and verify different parts of the implementation during the project.

The files in this folder are not required to reproduce the main thesis results, but they may be useful for understanding intermediate experiments and implementation choices.

## Requirements

The code is written in Julia. The main dependencies are listed in `Project.toml` and `Manifest.toml`.

The most important Julia packages used are:

- `VolumeFluxes.jl`
- `StaticArrays.jl`
- `ForwardDiff.jl`
- `Optim.jl`
- `Parameters.jl`
- `LinearAlgebra`
- `CairoMakie.jl`

In parts of the code, the package may still appear under the module name `SinFVM`, although the package/repository is now referred to as `VolumeFluxes.jl`.

To activate the project environment, run Julia from the repository root and use:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
