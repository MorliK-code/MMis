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

# Используем tasklist для получения всех процессов Python
$procs = tasklist /FO CSV /NH | ConvertFrom-Csv | Where-Object { $_.ImageName -like "*python*" }
foreach ($p in $procs) {
    $pid = [int]$p.PID
    if ($pid -le 0) { continue }
    if ($ExcludePid -gt 0 -and $pid -eq $ExcludePid) { continue }
    if ($pid -eq $PID) { continue }

    # Получаем командную строку через WMI
    $cmd = ""
    try {
        $wmiProc = Get-WmiObject Win32_Process -Filter "ProcessId = $pid" -ErrorAction Stop
        $cmd = [string]$wmiProc.CommandLine
    } catch {
        # Если WMI не сработал, пропускаем
        continue
    }
    
    if (-not $cmd) { continue }
    $cmdNorm = $cmd.ToLowerInvariant()

    # Проверяем тег
    $isTagged = $false
    if ($tagNorm) {
        if ($cmdNorm -match "--mmis-tag\s+[`"']?$([regex]::Escape($tagNorm))([`"']|\s|$)") {
            $isTagged = $true
        }
    }

    # Проверяем api_main.py
    $isApiMain = $cmdNorm.Contains("api_main.py")
    
    # Проверяем main.py --mode api
    $isMainApi = $cmdNorm.Contains("main.py") -and $cmdNorm.Contains("--mode") -and $cmdNorm.Contains("api")
    
    if (-not ($isTagged -or $isApiMain -or $isMainApi)) { continue }

    # Проверяем root если указан
    if ($rootNorm) {
        if ($cmdNorm.Contains($rootNorm) -eq $false -and $isTagged -eq $false) {
            continue
        }
    }

    # Завершаем процесс
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
