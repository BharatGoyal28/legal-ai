# Legal Contract Profiler — start both servers
# Usage: .\START.ps1 [your-gemini-api-key]

param([string]$GeminiKey = "")

if ($GeminiKey) {
    (Get-Content "backend\.env") -replace "GEMINI_API_KEY=.*", "GEMINI_API_KEY=$GeminiKey" |
        Set-Content "backend\.env"
    Write-Host "Gemini key saved to backend\.env"
}

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Host "`nStarting FastAPI backend on http://localhost:8001 ..."
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$PWD\backend'; python -m uvicorn main:app --host 0.0.0.0 --port 8001`""

Start-Sleep -Seconds 3

Write-Host "Starting Vite React dev server on http://localhost:5173 ..."
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$PWD\frontend'; npm run dev`""

Start-Sleep -Seconds 3

Write-Host "`nOpen http://localhost:5173 in your browser"
Write-Host "Backend health: http://localhost:8001/health"
