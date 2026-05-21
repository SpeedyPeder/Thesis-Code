# run_initialguess_6_7_pc.ps1

$ErrorActionPreference = "Stop"

# Folder containing this PowerShell file and the Julia script.
$ScriptDir = $PSScriptRoot

# Build output path from existing parent folders, avoiding manual Å encoding.
$ThesisCodeDir = Split-Path $ScriptDir -Parent
$MasterDir = Split-Path $ThesisCodeDir -Parent
$OutputDir = Join-Path $MasterDir "Optimization\2-D_Optimization\Initial Guess"

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

$LogFile = Join-Path $OutputDir "run_initialguess_cases_6_7_pc.log"

"Starting initial-guess cases 6 and 7 at $(Get-Date)" | Tee-Object -FilePath $LogFile -Append
"ScriptDir = $ScriptDir" | Tee-Object -FilePath $LogFile -Append
"OutputDir = $OutputDir" | Tee-Object -FilePath $LogFile -Append

function Run-JuliaCase($CaseNumber, $CaseName) {
    "Starting case $CaseNumber = $CaseName at $(Get-Date)" | Tee-Object -FilePath $LogFile -Append

    cmd /c "julia --project=.. .\2D-opt_initialguess_pc.jl $CaseNumber >> `"$LogFile`" 2>&1"

    if ($LASTEXITCODE -ne 0) {
        throw "Julia failed for case $CaseNumber = $CaseName with exit code $LASTEXITCODE. Check log: $LogFile"
    }

    "Finished case $CaseNumber = $CaseName at $(Get-Date)" | Tee-Object -FilePath $LogFile -Append
}

Run-JuliaCase 6 "shifted"
Run-JuliaCase 7 "random"

"Combining finished runs at $(Get-Date)" | Tee-Object -FilePath $LogFile -Append

cmd /c "julia --project=.. .\2D-opt_initialguess_pc.jl combine >> `"$LogFile`" 2>&1"

if ($LASTEXITCODE -ne 0) {
    throw "Julia combine failed with exit code $LASTEXITCODE. Check log: $LogFile"
}

"Finished all initial-guess case 6 and 7 runs at $(Get-Date)" | Tee-Object -FilePath $LogFile -Append