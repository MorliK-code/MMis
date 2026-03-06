param(
    [string]$Tag = "mmis",
    [string]$Root = "",
    [int]$ExcludePid = 0
)

$ErrorActionPreference = "SilentlyContinue"

$tagNorm = ($Tag | ForEach-Object { "$_".Trim().ToLowerInvariant() })
$rootNorm = ""
if ($Root) {
    try {
        $rootNorm = (Resolve-Path -LiteralPath $Root).Path.ToLowerInvariant()
    } catch {
        $rootNorm = "$Root".ToLowerInvariant()
    }
}

$killed = 0
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe' OR Name='py.exe'"
foreach ($p in $procs) {
    $pid = [int]$p.ProcessId
    if ($pid -le 0) { continue }
    if ($ExcludePid -gt 0 -and $pid -eq $ExcludePid) { continue }
    if ($pid -eq $PID) { continue }

    $cmd = [string]$p.CommandLine
    if (-not $cmd) { continue }
    $cmdNorm = $cmd.ToLowerInvariant()

    $isTagged = $false
    if ($tagNorm) {
        if ($cmdNorm -match "--mmis-tag\s+[`"']?$([regex]::Escape($tagNorm))([`"']|\s|$)") {
            $isTagged = $true
        }
    }

    $isApiMain = $cmdNorm.Contains("api_main.py")
    $isMainApi = $cmdNorm.Contains("main.py") -and $cmdNorm.Contains("--mode") -and $cmdNorm.Contains("api")
    if (-not ($isTagged -or $isApiMain -or $isMainApi)) { continue }

    if ($rootNorm) {
        if ($cmdNorm.Contains($rootNorm) -eq $false -and $isTagged -eq $false) {
            continue
        }
    }

    try {
        Stop-Process -Id $pid -Force -ErrorAction Stop
        $killed++
        Write-Output ("KILLED:{0}:{1}" -f $pid, $cmd)
    } catch {
        continue
    }
}

Write-Output ("KILLED_COUNT={0}" -f $killed)
exit 0
