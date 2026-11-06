$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")

# Example batch run with mock models
python -m bench.cli --model-id mock-main --judge-model-id mock-judge --split test --input-mode text

