param(
    [string]$OpenCV_DIR = $env:OpenCV_DIR,
    [switch]$BundledOpenCV,
    [ValidateRange(1, 64)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
$buildDir = Join-Path $projectDir 'build-windows'
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'Install CMake 3.24+ and Visual Studio 2022 C++ Build Tools, then use Developer PowerShell.'
}
& (Join-Path $PSScriptRoot 'download_models.ps1')
$cmakeArgs = @('-S', $projectDir, '-B', $buildDir, '-A', 'x64')
if ($BundledOpenCV) {
    $cmakeArgs += '-DBUILD_BUNDLED_OPENCV=ON'
} else {
    if (-not $OpenCV_DIR) { throw 'Pass -OpenCV_DIR C:\opencv\build or use -BundledOpenCV.' }
    $cmakeArgs += @('-DBUILD_BUNDLED_OPENCV=OFF', "-DOpenCV_DIR=$OpenCV_DIR")
}
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
& cmake --build $buildDir --config Release --target local_camera --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed.' }
# Official OpenCV Windows distributions store DLLs under x64/vcXX/bin.
if (-not $BundledOpenCV) {
    $dlls = @(Get-ChildItem -LiteralPath $OpenCV_DIR -Filter 'opencv_*.dll' -Recurse |
        Where-Object { $_.FullName -match '[\\/]x64[\\/]' -and $_.Name -notmatch '\dd\.dll$' })
    foreach ($dll in $dlls) { Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $buildDir 'bin') }
}
& ctest --test-dir $buildDir -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Runtime tests failed. Check OpenCV DLLs and models.' }
Write-Output "Run: $buildDir\bin\local_camera.exe"
