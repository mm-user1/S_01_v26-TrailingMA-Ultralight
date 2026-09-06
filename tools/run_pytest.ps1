[CmdletBinding()]
param(
    [switch]$KeepTemp,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = if ($env:MERLIN_PYTHON) {
    $env:MERLIN_PYTHON
} else {
    "C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe"
}

$RunnerArgs = @()
if ($KeepTemp) { $RunnerArgs += '--keep-temp' }
& $Python (Join-Path $RepoRoot 'tools\run_tests.py') @RunnerArgs -- @PytestArgs
exit $LASTEXITCODE
