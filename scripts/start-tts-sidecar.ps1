# Start the GPU TTS sidecar (port 8100).
# Requires .env with TTS_MODEL_ID, TTS_LOAD_IN_4BIT, TTS_USE_UNSLOTH.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Virtual env not found. Run: python -m venv .venv && pip install -r tts_server/requirements.txt"
}

.\.venv\Scripts\Activate.ps1

if (-not (Test-Path ".\.env")) {
    Write-Warning ".env missing — copy .env.example to .env and set TTS_MODEL_ID"
}

Write-Host "Starting TTS sidecar on http://0.0.0.0:8100 ..."
uvicorn tts_server.main:app --host 0.0.0.0 --port 8100
