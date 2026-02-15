<#
.SYNOPSIS
    Setup script for Emotion-Based ML Music Player.
    Downloads Python 3.11.7 (if needed), creates a virtual environment, and installs dependencies.

.DESCRIPTION
    This script automates the environment setup for the project when moving to a new PC.
    It checks for Python 3.11, downloads the official installer if missing, 
    sets up the venv, and installs all requirements.
#>

$PythonVersion = "3.11.7"
$PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
$InstallerPath = "$PSScriptRoot\python-installer.exe"
$VenvPath = "$PSScriptRoot\venv"

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   Moodify — Project Setup & Environment Builder" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# 1. Check if Python 3.11 is already installed
Write-Host "[1/4] Checking Python version..." -ForegroundColor Yellow
$py = Get-Command python.exe -ErrorAction SilentlyContinue
$isCorrectVersion = $false

if ($py) {
    $version = python --version
    if ($version -match "3.11") {
        Write-Host "✔ Found Python 3.11: $version" -ForegroundColor Green
        $isCorrectVersion = $true
    }
}

# 2. Download and Install Python if missing
if (-not $isCorrectVersion) {
    Write-Host "⚠ Python 3.11 not found. Downloading installer..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $InstallerPath
    
    Write-Host "⏳ Starting Python installation... Please follow the prompts." -ForegroundColor Cyan
    Write-Host "IMPORTANT: Ensure 'Add Python to PATH' is checked!" -ForegroundColor Red
    Start-Process -FilePath $InstallerPath -ArgumentList "/passive InstallAllUsers=1 PrependPath=1" -Wait
    
    Remove-Item $InstallerPath -ErrorAction SilentlyContinue
    
    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    $py = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Host "❌ Python installation failed or PATH not updated. Please install Python 3.11 manually." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit
    }
}

# 3. Create Virtual Environment
Write-Host "[2/4] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path $VenvPath) {
    Write-Host "✔ Virtual environment already exists." -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "✔ Virtual environment created." -ForegroundColor Green
}

# 4. Install Requirements
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r "$PSScriptRoot\requirements.txt"

# 5. Finalize
Write-Host "[4/4] Setup complete!" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "You can now run the project using:" -ForegroundColor White
Write-Host "  - .\Emotion Based ML Music Player\run_main.ps1" -ForegroundColor Green
Write-Host "  - .\Emotion Based ML Music Player\run_live_detector.ps1" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan

Read-Host "Press Enter to exit"
