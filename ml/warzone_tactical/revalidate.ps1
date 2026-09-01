param(
    [int]$Battles = 100,
    [int]$GoldenRequired = 10000,
    [string]$Output = "ml\warzone_tactical\runs\revalidation-current",
    [string]$TrainedManifest = "ml\warzone_tactical\runs\stage1-promotion-rocket\seed-4\artifacts\policy.manifest.json"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$executable = Join-Path $repository "build_win\src\warzone2100.exe"

if (-not (Test-Path -LiteralPath $cmake)) {
    throw "Visual Studio CMake was not found at $cmake"
}

& $cmake --build (Join-Path $repository "build_win") --config Release --target warzone2100 --parallel 12
if ($LASTEXITCODE -ne 0) {
    throw "The ML engine build failed."
}

$arguments = @(
    "run", "--project", (Join-Path $repository "ml\warzone_tactical"),
    "python", "-m", "warzone_tactical.revalidate",
    "--executable", $executable,
    "--repository", $repository,
    "--output", (Join-Path $repository $Output),
    "--battles", $Battles,
    "--golden-required", $GoldenRequired
)
if ($TrainedManifest -and (Test-Path -LiteralPath (Join-Path $repository $TrainedManifest))) {
    $arguments += @("--trained-manifest", (Join-Path $repository $TrainedManifest))
}
& uv @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Contract V1 revalidation failed."
}
