[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AssetDirectory,

    [string]$Destination = ".\auction-moment-archive-v1",

    [string]$PythonExecutable = "",

    [switch]$SkipRuntimeDependencies
)

$ErrorActionPreference = "Stop"

$assets = (Resolve-Path -LiteralPath $AssetDirectory).Path
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$environmentPath = Join-Path $destinationPath ".venv"
$snapshotPath = Join-Path $destinationPath "snapshot"
$checksumPath = Join-Path $assets "SHA256SUMS.txt"

if (Test-Path -LiteralPath $destinationPath) {
    throw "Destination already exists: $destinationPath"
}
if (-not (Test-Path -LiteralPath $checksumPath -PathType Leaf)) {
    throw "Missing SHA256SUMS.txt in $assets"
}

Write-Host "Verifying archive Release assets"
foreach ($line in Get-Content -LiteralPath $checksumPath) {
    if ($line -notmatch '^([0-9a-f]{64})  (.+)$') {
        throw "Malformed checksum line: $line"
    }
    $expected = $Matches[1]
    $name = $Matches[2]
    $path = Join-Path $assets $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing checksummed asset: $name"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "SHA256 mismatch: $name"
    }
}

New-Item -ItemType Directory -Path $destinationPath | Out-Null
Expand-Archive -LiteralPath (Join-Path $assets "auction-moment-public-archive-v1.zip") -DestinationPath $snapshotPath

Write-Host "Creating CPython 3.12 environment"
if ($PythonExecutable) {
    $resolvedPython = (Resolve-Path -LiteralPath $PythonExecutable).Path
    & $resolvedPython -m venv $environmentPath
} else {
    & py -3.12 -m venv $environmentPath
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the Python 3.12 environment"
}
$python = Join-Path $environmentPath "Scripts\python.exe"
& $python -m pip install pip==26.2.1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the frozen pip version"
}

if (-not $SkipRuntimeDependencies) {
    & $python -m pip install -r (Join-Path $assets "requirements-runtime-win-py312.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install frozen runtime dependencies"
    }
}

$researchWheel = Join-Path $assets "auction_moment_research-0.1.0-py3-none-any.whl"
$assistantWheel = Join-Path $assets "auction_moment_assistant-0.2.0b6-py3-none-any.whl"
& $python -m pip install --no-deps $researchWheel $assistantWheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the archived project wheels"
}
if (-not $SkipRuntimeDependencies) {
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "The restored environment failed pip check"
    }
}

$assistant = Join-Path $environmentPath "Scripts\auction-vision-assistant.exe"
$treasures = Join-Path $snapshotPath "data\v1\core\treasures.csv"
Write-Host "Archive restored and verified. The project remains hibernating."
Write-Host "After completing RETURN_RUNBOOK.md, the experimental entry point would be:"
Write-Host "  $assistant --models $assets --treasures $treasures"
