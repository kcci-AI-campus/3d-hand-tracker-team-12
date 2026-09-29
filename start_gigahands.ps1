$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$simPython = Get-Command python -ErrorAction SilentlyContinue
if ($simPython -and $simPython.Source -notlike '*WindowsApps*') {
    & $simPython.Source (Join-Path $PSScriptRoot 'gigahands_sim.py') @args
    exit $LASTEXITCODE
}
$simRuntime = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python'
$simExecutable = Join-Path $simRuntime 'python.exe'
if (-not (Test-Path -LiteralPath $simExecutable)) {
    throw 'Install Python 3.10+ with Tcl/Tk, then run: python -m pip install -r requirements-simulator.txt'
}
# The bundled runtime Tcl path may be unavailable to Tcl's native file resolver.
$simTcl = Join-Path $PSScriptRoot '.tools\simulator-tcl'
if (-not (Test-Path -LiteralPath (Join-Path $simTcl 'tcl8.6\init.tcl'))) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '.tools') | Out-Null
    Copy-Item -LiteralPath (Join-Path $simRuntime 'tcl') -Destination $simTcl -Recurse
}
$simOldTcl = $env:TCL_LIBRARY
$simOldTk = $env:TK_LIBRARY
try {
    $env:TCL_LIBRARY = (Join-Path $simTcl 'tcl8.6').Replace('\', '/')
    $env:TK_LIBRARY = (Join-Path $simTcl 'tk8.6').Replace('\', '/')
    & $simExecutable (Join-Path $PSScriptRoot 'gigahands_sim.py') @args
    $simExit = $LASTEXITCODE
} finally {
    $env:TCL_LIBRARY = $simOldTcl
    $env:TK_LIBRARY = $simOldTk
}
exit $simExit
