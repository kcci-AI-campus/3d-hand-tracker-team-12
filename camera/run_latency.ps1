$ErrorActionPreference = 'Stop'
# Run from the repository root so logs land in <repo>/latency_logs.
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$latencyPython = Get-Command python -ErrorAction SilentlyContinue
if ($latencyPython -and $latencyPython.Source -notlike '*WindowsApps*') {
    $latencyExecutable = $latencyPython.Source
} else {
    $latencyExecutable = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
}
if (-not (Test-Path -LiteralPath $latencyExecutable)) {
    throw 'Python 3.10+ is required. The PC receiver uses only the Python standard library.'
}
if ($args.Count -eq 0) {
    & $latencyExecutable (Join-Path $PSScriptRoot 'latency_recorder.py') receiver
} else {
    & $latencyExecutable (Join-Path $PSScriptRoot 'latency_recorder.py') @args
}
exit $LASTEXITCODE
