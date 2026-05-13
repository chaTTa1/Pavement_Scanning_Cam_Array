param(
    [Parameter(Mandatory = $true)]
    [string]$Port,

    [int]$Baud = 115200,

    [double]$Duration = 10,

    [string]$OutPrefix = ("mip_capture_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
)

function Get-Fletcher16 {
    param([byte[]]$Data)

    [int]$c0 = 0
    [int]$c1 = 0
    foreach ($b in $Data) {
        $c0 = ($c0 + [int]$b) -band 0xFF
        $c1 = ($c1 + $c0) -band 0xFF
    }
    return @($c0, $c1)
}

function Find-MipSync {
    param([System.Collections.Generic.List[byte]]$Buffer)

    for ($i = 0; $i -lt ($Buffer.Count - 1); $i++) {
        if ($Buffer[$i] -eq 0x75 -and $Buffer[$i + 1] -eq 0x65) {
            return $i
        }
    }
    return -1
}

function Get-MipFieldsText {
    param([byte[]]$Payload)

    $fields = New-Object System.Collections.Generic.List[string]
    $i = 0
    while (($i + 2) -le $Payload.Length) {
        $fieldLen = [int]$Payload[$i]
        if ($fieldLen -lt 2 -or ($i + $fieldLen) -gt $Payload.Length) {
            break
        }
        $fieldDesc = [int]$Payload[$i + 1]
        $fields.Add(("0x{0:X2}:{1}" -f $fieldDesc, ($fieldLen - 2)))
        $i += $fieldLen
    }
    return ($fields -join " ")
}

$rawPath = "$OutPrefix.bin"
$csvPath = "$OutPrefix.csv"
$buffer = New-Object System.Collections.Generic.List[byte]
$rows = New-Object System.Collections.Generic.List[object]
$readBuf = New-Object byte[] 4096
$packetIndex = 0
$badChecksums = 0

$serial = [System.IO.Ports.SerialPort]::new($Port, $Baud, [System.IO.Ports.Parity]::None, 8, [System.IO.Ports.StopBits]::One)
$serial.ReadTimeout = 100
$raw = [System.IO.File]::Create((Resolve-Path -LiteralPath ".").Path + [System.IO.Path]::DirectorySeparatorChar + $rawPath)

try {
    $serial.Open()
    $deadline = [DateTime]::UtcNow.AddSeconds($Duration)

    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $n = $serial.Read($readBuf, 0, $readBuf.Length)
        } catch [TimeoutException] {
            continue
        }

        if ($n -le 0) {
            continue
        }

        $raw.Write($readBuf, 0, $n)
        for ($i = 0; $i -lt $n; $i++) {
            $buffer.Add($readBuf[$i])
        }

        while ($true) {
            $syncPos = Find-MipSync -Buffer $buffer
            if ($syncPos -lt 0) {
                if ($buffer.Count -gt 1) {
                    $buffer.RemoveRange(0, $buffer.Count - 1)
                }
                break
            }
            if ($syncPos -gt 0) {
                $buffer.RemoveRange(0, $syncPos)
            }
            if ($buffer.Count -lt 6) {
                break
            }

            $payloadLen = [int]$buffer[3]
            $packetLen = 6 + $payloadLen
            if ($buffer.Count -lt $packetLen) {
                break
            }

            $packet = New-Object byte[] $packetLen
            for ($i = 0; $i -lt $packetLen; $i++) {
                $packet[$i] = $buffer[$i]
            }
            $buffer.RemoveRange(0, $packetLen)

            $checkData = New-Object byte[] ($packetLen - 4)
            [Array]::Copy($packet, 2, $checkData, 0, $packetLen - 4)
            $calc = Get-Fletcher16 -Data $checkData
            $expected0 = [int]$packet[$packetLen - 2]
            $expected1 = [int]$packet[$packetLen - 1]
            $ok = ($calc[0] -eq $expected0 -and $calc[1] -eq $expected1)
            if (-not $ok) {
                $badChecksums++
            }

            $payload = New-Object byte[] $payloadLen
            if ($payloadLen -gt 0) {
                [Array]::Copy($packet, 4, $payload, 0, $payloadLen)
            }
            $fieldsText = Get-MipFieldsText -Payload $payload
            $fieldCount = if ($fieldsText.Length -eq 0) { 0 } else { ($fieldsText -split " ").Count }

            $rows.Add([pscustomobject]@{
                wall_time = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
                packet_index = $packetIndex
                desc_set = ("0x{0:X2}" -f [int]$packet[2])
                payload_len = $payloadLen
                checksum_ok = if ($ok) { "yes" } else { "no" }
                checksum_expected = ("0x{0:X2} 0x{1:X2}" -f $expected0, $expected1)
                checksum_actual = ("0x{0:X2} 0x{1:X2}" -f $calc[0], $calc[1])
                field_count = $fieldCount
                fields = $fieldsText
            })
            $packetIndex++
        }
    }
} finally {
    $raw.Close()
    if ($serial.IsOpen) {
        $serial.Close()
    }
}

$rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation
Write-Host "Raw capture: $rawPath"
Write-Host "CSV summary: $csvPath"
Write-Host "Packets: $packetIndex"
Write-Host "Bad checksums: $badChecksums"

if ($packetIndex -eq 0 -or $badChecksums -ne 0) {
    exit 1
}
