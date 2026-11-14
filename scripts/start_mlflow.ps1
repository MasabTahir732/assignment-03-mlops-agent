$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
New-Item -ItemType Directory -Force reports/mlflow-artifacts | Out-Null
$artifacts = (Resolve-Path reports/mlflow-artifacts).Path
& $python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --serve-artifacts --artifacts-destination $artifacts
