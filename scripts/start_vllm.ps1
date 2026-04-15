$ErrorActionPreference = "Stop"

param(
  [string]$Model = "Qwen/Qwen2.5-7B-Instruct",
  [int]$Port = 8001,
  [int]$TensorParallel = 1,
  [string]$GpuIds = ""
)

if ($GpuIds -ne "") {
  $env:CUDA_VISIBLE_DEVICES = $GpuIds
}

# Multi-GPU example: -TensorParallel 2 -GpuIds "0,1"
python -m vllm.entrypoints.openai.api_server `
  --model $Model `
  --port $Port `
  --tensor-parallel-size $TensorParallel

