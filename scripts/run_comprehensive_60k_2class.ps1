$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$config = Join-Path $projectRoot 'configs\comprehensive_60k_2class.yml'
$dataset = 'F:\MapleStoryAssets\datasets\mxdclassic_monster_player_1280x224\comprehensive_60k_2class'

Set-Location -LiteralPath $projectRoot
& $python -u -m classic_dataset.comprehensive --config $config
if ($LASTEXITCODE -ne 0) {
    throw "60K two-class generation failed with exit code $LASTEXITCODE"
}

& $python -u -m classic_dataset.validate_comprehensive --config $config --dataset $dataset
if ($LASTEXITCODE -ne 0) {
    throw "60K two-class validation failed with exit code $LASTEXITCODE"
}
