$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PSScriptRoot
& ..\venv\Scripts\Activate.ps1
python "main.py"
Read-Host -Prompt "Press Enter to exit"
