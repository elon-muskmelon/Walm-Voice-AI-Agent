# Start the main voice pipeline app (port 8000).
# Requires .env with GEMINI_API_KEY, ELEVENLABS_API_KEY, TTS_SERVICE_URL.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Virtual env not found. Run: python -m venv .venv && pip install -r requirements.txt"
}

.\.venv\Scripts\Activate.ps1

if (-not (Test-Path ".\.env")) {
    Write-Error ".env missing — copy .env.example to .env and set API keys"
}

Write-Host "Starting main app on http://0.0.0.0:8000 ..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
