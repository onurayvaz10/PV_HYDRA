# Climate expansion on the local GPU machine (Windows PowerShell). Same order as run_climate_expansion_local.sh;
# every step is resumable, so the script can simply be restarted after an interruption.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PY) { $env:PY } else { "python" }
$steps = @(
  "scripts/fetch_climate_expansion.py",
  "scripts/screen_climate_expansion.py",
  "scripts/run_climate_expansion.py queue",
  "scripts/run_semisynthetic_climate.py weather",
  "scripts/run_semisynthetic_climate.py queue",
  "scripts/run_semisynthetic_climate.py cpu",
  "scripts/run_semisynthetic_climate.py gpu",
  "scripts/run_climate_expansion.py rd",
  "scripts/run_climate_expansion.py kappa",
  "scripts/run_climate_expansion.py ml",
  "scripts/run_semisynthetic_climate.py summary",
  "scripts/summarize_climate_expansion.py"
)
foreach ($step in $steps) {
  Write-Host ">> $py $step"
  & $py @($step -split " ")
  if ($LASTEXITCODE -ne 0) { throw "failed: $step" }
}
