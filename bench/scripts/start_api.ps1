$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

python -m uvicorn bench.api.server:app --host 127.0.0.1 --port 8010 --reload

