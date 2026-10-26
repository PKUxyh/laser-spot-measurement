$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildRoot = Join-Path $projectRoot '.build'
$distRoot = Join-Path $projectRoot 'release'
$portableRoot = Join-Path $distRoot 'FocusAnalysisPortable'
$zipPath = Join-Path $distRoot 'FocusAnalysisPortable-win64.zip'
$python = Join-Path $buildRoot 'venv\Scripts\python.exe'

Set-Location $projectRoot
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .build\venv with Python 3.10 and install requirements-build.txt first; see README.md.'
}
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $python -m pytest tests --assert=plain -q
if ($LASTEXITCODE -ne 0) {
    throw 'Tests failed; release build stopped.'
}
& $python -m PyInstaller --noconfirm --clean `
    --workpath (Join-Path $buildRoot 'pyinstaller') `
    --distpath $buildRoot `
    FocusAnalysis.spec
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller build failed.'
}

$exePath = Join-Path $buildRoot 'FocusAnalysis\FocusAnalysis.exe'
$smokeReport = Join-Path $buildRoot 'smoke-test.json'
if (Test-Path -LiteralPath $smokeReport) {
    Remove-Item -LiteralPath $smokeReport
}
$smokeProcess = Start-Process -FilePath $exePath -ArgumentList @('--self-test', ('"' + $smokeReport + '"')) -WindowStyle Hidden -PassThru
if (-not $smokeProcess.WaitForExit(240000)) {
    $smokeProcess.Kill()
    throw 'Frozen GUI smoke test timed out.'
}
if ($smokeProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $smokeReport)) {
    throw 'Frozen GUI smoke test failed; inspect .build\smoke-test.json.'
}
if (-not (Get-Content -LiteralPath $smokeReport -Raw -Encoding UTF8 | ConvertFrom-Json).passed) {
    throw 'Frozen GUI smoke test reported a failure.'
}

if (Test-Path -LiteralPath $portableRoot) {
    $resolvedPortable = (Resolve-Path -LiteralPath $portableRoot).Path
    $resolvedProject = (Resolve-Path -LiteralPath $projectRoot).Path
    if ($resolvedPortable -ne (Join-Path $resolvedProject 'release\FocusAnalysisPortable')) {
        throw 'Refusing to remove a directory outside the expected release location.'
    }
    Remove-Item -LiteralPath $portableRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $portableRoot | Out-Null
Copy-Item -Path (Join-Path $buildRoot 'FocusAnalysis\*') `
    -Destination $portableRoot -Recurse -Force
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'packaging') -Filter '*.txt' | `
    Copy-Item -Destination $portableRoot

$sourceRoot = Join-Path $portableRoot 'source'
New-Item -ItemType Directory -Path $sourceRoot | Out-Null
$sourceFiles = @(
    'analysis.py', 'app_gui.py', 'batch_analysis.py', 'data_io.py',
    'zernike_analysis.py', 'zernike_rzern.py', 'README.md',
    'requirements.txt', 'requirements-build.txt', 'FocusAnalysis.spec',
    'build_release.ps1'
)
foreach ($sourceFile in $sourceFiles) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $sourceFile) `
        -Destination $sourceRoot
}
$sourceTests = Join-Path $sourceRoot 'tests'
New-Item -ItemType Directory -Path $sourceTests | Out-Null
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'tests') -Filter '*.py' | `
    Copy-Item -Destination $sourceTests
$sourcePackaging = Join-Path $sourceRoot 'packaging'
New-Item -ItemType Directory -Path $sourcePackaging | Out-Null
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'packaging') -File | `
    Copy-Item -Destination $sourcePackaging
Copy-Item -LiteralPath $smokeReport -Destination (Join-Path $portableRoot 'verification.json')
& $python -m pip freeze | Set-Content -LiteralPath (Join-Path $sourceRoot 'requirements-lock.txt') -Encoding UTF8

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -LiteralPath $portableRoot -DestinationPath $zipPath `
    -CompressionLevel Optimal
$checksum = Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath
($checksum.Hash + '  ' + (Split-Path -Leaf $zipPath)) | `
    Set-Content -LiteralPath ($zipPath + '.sha256') -Encoding ASCII
$checksum
