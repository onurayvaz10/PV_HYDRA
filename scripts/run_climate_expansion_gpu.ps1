# GPU part of the climate expansion (Windows PowerShell): kappa-HYDRA-D, lstm, gru, tcn, patchtst, informer.
# See run_climate_expansion_gpu.sh for prerequisites. Resumable; expected ~11 h on an RTX 4080.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PY) { $env:PY } else { "python" }
$nets = "lstm,gru,tcn,patchtst,informer"
$env:RUN_TAG = "_gpu"
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA device')"
if ($LASTEXITCODE -ne 0) { throw "no CUDA device" }
$steps = @(
  @("scripts/fetch_climate_expansion.py", "--selected"),
  (@("scripts/run_semisynthetic_climate.py", "gpu", "khd_full") + $nets.Split(",")),
  @("scripts/run_climate_expansion.py", "kappa"),
  @("scripts/run_climate_expansion.py", "ml", "--models=$nets")
)
foreach ($step in $steps) {
  Write-Host ">> $py $($step -join ' ')"
  & $py @step
  if ($LASTEXITCODE -ne 0) { throw "failed: $($step -join ' ')" }
}
Write-Host "done. Commit only: results/climate_expansion/semisynthetic_climate_runs_gpu.csv, rates_kappa_gpu.csv, rates_ml_gpu.csv"
