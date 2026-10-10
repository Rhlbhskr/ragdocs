$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& "$PSScriptRoot/.venv/Scripts/python.exe" -m streamlit run streamlit_app/app.py
