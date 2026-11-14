param([switch]$SkipInstall)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:DVC_SITE_CACHE_DIR = Join-Path (Get-Location) '.dvc/tmp/site-cache'
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE)" } }
if (!(Test-Path '.venv/Scripts/python.exe')) { python -m venv .venv; Check-Exit }
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (!$SkipInstall) { & $python -m pip install -e '.[dev]'; Check-Exit }
if (!(Test-Path '.git')) { git init; Check-Exit }
if (!(Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    foreach ($key in @('AIRFLOW_DB_PASSWORD', 'AIRFLOW_SECRET_KEY', 'AIRFLOW_JWT_SECRET')) {
        $bytes = New-Object byte[] 24
        $rng.GetBytes($bytes)
        $secret = ($bytes | ForEach-Object { $_.ToString('x2') }) -join ''
        Add-Content '.env' "$key=$secret"
    }
    $rng.Dispose()
}
if (!(Test-Path '.dvc')) { & $python -m dvc init; Check-Exit }
if (!(Test-Path '.dvc/config.local')) {
    & $python -m dvc remote add --local -d local-store (Join-Path (Get-Location) '.dvc-remote')
    Check-Exit
}
$hasV1 = @(git tag --list v1).Count -gt 0
Check-Exit
if (!(Test-Path 'data/iris.csv.dvc') -or !$hasV1) {
    & $python -m training.train --generate v1; Check-Exit
    & $python -m dvc add data/iris.csv; Check-Exit
    & $python -m dvc push; Check-Exit
    git add .; Check-Exit
    git commit -m 'Implement MLOps agent and small Iris v1 dataset'; Check-Exit
    git tag v1; Check-Exit
    & $python -m training.train --generate v2; Check-Exit
    & $python -m dvc add data/iris.csv; Check-Exit
    & $python -m dvc push; Check-Exit
    git add data/iris.csv.dvc; Check-Exit
    git commit -m 'Version tiny Iris dataset v2 (150 rows)'; Check-Exit
    git tag v2; Check-Exit
} elseif (!(Test-Path '.dvc-remote')) {
    # A fresh GitHub clone has DVC pointers, not cached CSV files. Recreate both tiny objects.
    $original = Get-Content 'data/iris.csv.dvc' -Raw
    foreach ($version in @('v1', 'v2')) {
        & $python -m training.train --generate $version; Check-Exit
        & $python -m dvc add data/iris.csv; Check-Exit
        & $python -m dvc push; Check-Exit
    }
    if ((Get-Content 'data/iris.csv.dvc' -Raw) -ne $original) {
        throw 'Regenerated v2 pointer differs from the checked-in pointer. Inspect before training.'
    }
}
Write-Host 'Setup ready. Dataset stays under 5 KB. Follow README for service startup.'
