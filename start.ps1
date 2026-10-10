$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    & $pythonPath -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install dependencies.' }
}
Write-Host 'GroundedDocs: http://localhost:8000'
Write-Host 'API documentation: http://localhost:8000/docs'
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000
