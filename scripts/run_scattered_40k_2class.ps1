$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

& '.\.venv\Scripts\python.exe' -u -m classic_dataset.scattered_formal `
  --config 'configs\scattered_40k_2class.yml'

if ($LASTEXITCODE -ne 0) {
    throw "40K dataset generation failed with exit code $LASTEXITCODE"
}

