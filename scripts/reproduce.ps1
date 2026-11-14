param([ValidateSet('v1', 'v2')][string]$Revision = 'v1')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$root = (Get-Location).Path
$env:DVC_SITE_CACHE_DIR = Join-Path $root '.dvc/tmp/site-cache'
$python = Join-Path $root '.venv/Scripts/python.exe'
$destination = Join-Path $root "reports/reproduce-$Revision"
if (Test-Path $destination) { throw 'Reproduction destination exists; choose a fresh clone for another run.' }
git worktree add --detach $destination $Revision
if ($LASTEXITCODE -ne 0) { throw 'Git worktree failed' }
Push-Location $destination
try {
    & $python -m dvc remote add --local -d local-store (Join-Path $root '.dvc-remote')
    if ($LASTEXITCODE -ne 0) { throw 'Remote configuration failed' }
    & $python -m dvc pull data/iris.csv.dvc
    if ($LASTEXITCODE -ne 0) { throw 'DVC pull failed' }
    & $python -c "import pandas as pd; print('Reproduced rows:', len(pd.read_csv('data/iris.csv')))"
    Get-FileHash data/iris.csv -Algorithm MD5
} finally { Pop-Location }
