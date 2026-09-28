$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
$modelDir = Join-Path $projectDir 'models'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
$base = 'https://raw.githubusercontent.com/FeiGeChuanShu/ncnn-Android-mediapipe_hand/65b1bcac06c5b9a1a492be3afebc9e4d52680f5c/app/src/main/assets'
$hashes = @{
    'palm-lite-op.param' = '2ee6bd3af90098697f0cb81466afff37833cfda3ae1f3b9a57dbf6508a9532ca'
    'palm-lite-op.bin' = 'ad5b7e2f8083084ce2b6c7825f5a809b7b9e576cde60ce4cfdc600fd28cc40f4'
    'hand_lite-op.param' = '93a7fd606639bbec99d70d05b57651b0bd34fe760b116d313b7048cc751af33e'
    'hand_lite-op.bin' = '58480bf634e8761192c056cfb952e62b721b4661e347227069dd56d94ca3a981'
}
foreach ($name in $hashes.Keys) {
    $target = Join-Path $modelDir $name
    if ((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $hashes[$name]) {
        Write-Output "Verified: $name"
        continue
    }
    $partial = "$target.part"
    try {
        Invoke-WebRequest "$base/$name" -OutFile $partial -UseBasicParsing
        if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $hashes[$name]) {
            throw "Model checksum mismatch: $name"
        }
        Move-Item -LiteralPath $partial -Destination $target -Force
    } finally {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
    }
}
