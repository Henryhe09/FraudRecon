$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8010 --reload

