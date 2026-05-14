<#
.SYNOPSIS
    Record MIP binary stream from Teensy 4.1 USB serial port.

.PARAMETER ComPort
    Serial port name (default: COM12)

.PARAMETER BaudRate
    Baud rate (default: 115200)

.PARAMETER DurationSeconds
    Recording duration in seconds (default: 20)

.PARAMETER OutputDir
    Output directory (default: ./recordings next to this script)

.EXAMPLE
    .\record_mip.ps1
    .\record_mip.ps1 -ComPort COM5 -DurationSeconds 30
#>

param(
    [string]$ComPort = "COM12",
    [int]$BaudRate = 115200,
    [int]$DurationSeconds = 20,
    [string]$OutputDir = $null
)

# Resolve output directory relative to script
if ([string]::IsNullOrEmpty($OutputDir)) {
    $OutputDir = Join-Path $PSScriptRoot "recordings"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFileAbs = Join-Path (Resolve-Path $OutputDir).Path "mip_capture_$timestamp.bin"

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " MIP Stream Recorder" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " Port             : $ComPort"
Write-Host " Baud             : $BaudRate"
Write-Host " Duration         : $DurationSeconds seconds"
Write-Host " Output file      : $outputFileAbs"
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Open serial port
Write-Host "[INFO] Opening $ComPort ..." -ForegroundColor Yellow
try {
    $port = New-Object System.IO.Ports.SerialPort $ComPort, $BaudRate, 'None', 8, 'One'
    $port.ReadTimeout = 50          # shorter timeout, more responsive
    $port.WriteTimeout = 500
    $port.DtrEnable = $true
    $port.RtsEnable = $true
    $port.Open()
}
catch {
    Write-Host "[ERROR] Failed to open ${ComPort}: $_" -ForegroundColor Red
    Write-Host "Check that:" -ForegroundColor Yellow
    Write-Host "  - Teensy is plugged in"
    Write-Host "  - No other program (Serial Monitor, etc.) is using the port"
    Write-Host "  - You have permission to access the port"
    exit 1
}

Write-Host "[INFO] Recording in progress, press Ctrl+C to abort early..." -ForegroundColor Yellow
Write-Host ""

# Open output stream
$fileStream = [System.IO.File]::Open($outputFileAbs, 'Create')

$startTime = Get-Date
$endTime = $startTime.AddSeconds($DurationSeconds)
$totalBytes = 0
$lastReport = $startTime
$buffer = New-Object byte[] 4096

try {
    while ((Get-Date) -lt $endTime) {
        try {
            $available = $port.BytesToRead
        }
        catch {
            $available = 0
        }

        if ($available -gt 0) {
            $toRead = [Math]::Min($available, $buffer.Length)
            try {
                $n = $port.Read($buffer, 0, $toRead)
                if ($n -gt 0) {
                    $fileStream.Write($buffer, 0, $n)
                    $totalBytes += $n
                }
            }
            catch [System.TimeoutException] {
                # Read timed out, just continue
            }
        }
        else {
            Start-Sleep -Milliseconds 1    # 1ms instead of 5ms
        }

        # Progress report every 1 second
        $now = Get-Date
        if (($now - $lastReport).TotalSeconds -ge 1.0) {
            $elapsed = ($now - $startTime).TotalSeconds
            $avgRate = if ($elapsed -gt 0) { [int]($totalBytes / $elapsed) } else { 0 }
            Write-Host ("[t={0,5:F1}s] Recorded {1,7} bytes (avg {2,6} B/s)" -f $elapsed, $totalBytes, $avgRate)
            $lastReport = $now
        }
    }
}
catch {
    Write-Host ""
    Write-Host "[INFO] Recording stopped early: $($_.Exception.Message)" -ForegroundColor Yellow
}
finally {
    Write-Host ""
    Write-Host "[INFO] Closing port and file..." -ForegroundColor Yellow
    try { $fileStream.Flush() } catch {}
    try { $fileStream.Close() } catch {}
    try { $port.Close() } catch {}
    try { $port.Dispose() } catch {}
}

$elapsedTotal = ((Get-Date) - $startTime).TotalSeconds
$avgRateTotal = if ($elapsedTotal -gt 0) { [int]($totalBytes / $elapsedTotal) } else { 0 }

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " SUMMARY" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " Bytes recorded   : $totalBytes"
Write-Host " Duration         : $($elapsedTotal.ToString('F2')) s"
Write-Host " Average rate     : $avgRateTotal bytes/sec"
Write-Host " Output file      : $outputFileAbs"

if ($totalBytes -eq 0) {
    Write-Host "[WARN] No data received! Check:" -ForegroundColor Red
    Write-Host "  - MIRROR_MIP_TO_USB = true in main.cpp"
    Write-Host "  - simpleRTK3B is connected and outputting SBF"
    Write-Host "  - GNSS has at least Stand-Alone fix (mode >= 1)"
}
elseif ($totalBytes -lt 1000) {
    Write-Host "[WARN] Very little data ($totalBytes B), possibly no GNSS fix" -ForegroundColor Yellow
}
elseif ($totalBytes -gt 100000) {
    Write-Host "[WARN] More data than expected" -ForegroundColor Yellow
    Write-Host "       Status text might be mixed in. Check MIRROR_MIP_TO_USB / PRINT_STATUS_TO_USB"
}
else {
    Write-Host "[OK] File looks healthy" -ForegroundColor Green
}
Write-Host "======================================================" -ForegroundColor Cyan