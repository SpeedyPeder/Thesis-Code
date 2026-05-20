# run_initialguess_remaining_pc.ps1
# Runs the remaining initial-guess cases (4--7) on your Windows PC.
# Your PC must stay powered on and awake. This will keep running if VS Code is closed.

$ErrorActionPreference = "Stop"

# Folder containing the Julia script and the parent Project.toml environment.
$ScriptDir = "C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Thesis Code\Optimization_tests"

# Output folder for figures and text files.
$OutputDir = "C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization\2-D_Optimization\Initial Guess"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Set-Location $ScriptDir

$env:INITIAL_GUESS_OUTPUT_DIR = $OutputDir
$env:JULIA_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

$env:LBFGS_MAX_ITERS = "6"
$env:RUN_GN_AFTER_LBFGS = "false"
$env:GN_AFTER_LBFGS_ITERS = "1"
$env:USE_FULL_GUESS_SET = "true"
$env:MAKE_PLOTS = "true"
$env:SKIP_EXISTING = "true"

$LogFile = Join-Path $OutputDir "run_initialguess_remaining_pc.log"

"Starting remaining initial-guess runs at $(Get-Date)" | Tee-Object -FilePath $LogFile -Append
"ScriptDir = $ScriptDir" | Tee-Object -FilePath $LogFile -Append
"OutputDir = $OutputDir" | Tee-Object -FilePath $LogFile -Append

# --project=.. assumes Optimization_tests is inside the thesis project folder.
julia --project=.. .\2D-opt_initialguess_pc.jl remaining *>> $LogFile

"Finished remaining initial-guess runs at $(Get-Date)" | Tee-Object -FilePath $LogFile -Append