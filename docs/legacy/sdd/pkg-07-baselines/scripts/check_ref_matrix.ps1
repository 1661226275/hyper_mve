# check_ref_matrix.ps1 -- Pkg-07 SDD reference-matrix machine check (Day 8 per pkg-07 README calendar)
#
# Usage:
#     cd D:\RL\hyper_mve\sdd\pkg-07-baselines
#     pwsh scripts\check_ref_matrix.ps1        # also runs under Windows PowerShell 5.1
#
# Output:
#     ref_matrix.csv  with columns: spec / refs_actual / refs_expected / delta
#
# Acceptance:
#     delta column all empty (each spec's actual refs are a superset of README "spec 间引用表" expected refs)
#
# NOTE: ASCII-only console output/comments on purpose -- .ps1 is UTF-8 without BOM, and
# Windows PowerShell 5.1 parses scripts in the ANSI codepage. CJK text here would mojibake
# and break the parser. Keep this file ASCII so it runs under any PowerShell edition.

$ErrorActionPreference = "Stop"

# README "spec 间引用表" expected reference matrix (hand-maintained; must stay in sync with README.md lines 285-292)
$ExpectedRefs = @{
    "01-baseline-registry-and-cli"          = @("02", "03", "04", "05", "06", "08")
    "02-shared-backbones-internal"          = @("03", "08")
    "03-internal-variants"                  = @("02", "07", "08")
    "04-pettingzoo-adapter"                 = @("05", "06", "08")
    "05-external-mappo"                     = @("04", "07", "08")
    "06-external-qmix-mamuzero-mamba"       = @("04", "07", "08")
    "07-fairness-protocol"                  = @("03", "05", "06", "08")
    "08-integration-contracts"              = @("01", "02", "03", "04", "05", "06", "07")
}

# Scan each spec file
$Results = @()

Get-ChildItem "specs\0*.md" | Sort-Object Name | ForEach-Object {
    $name = $_.BaseName
    $content = Get-Content $_.FullName -Raw

    # Match cross-spec references in two forms, and exclude cross-PACKAGE refs:
    #   - bare "spec 0X" or "spec 0X.Y"     but NOT "pkg-NN spec 0X" or "Pkg-NN spec 0X"
    #     (case-insensitive negative lookbehind, per spec 08 #6.2 lock)
    #   - filename "0X-name" (how specs cite each other in the cross-ref table)
    # This produces intra-package refs only.
    $pat = '(?<![Pp]kg-\d{2} )spec\s+0([1-8])|\b0([1-8])-[a-z]'
    $refs = [regex]::Matches($content, $pat) |
            ForEach-Object { if ($_.Groups[1].Value) { $_.Groups[1].Value } else { $_.Groups[2].Value } } |
            Sort-Object -Unique

    # Two-digit zero-padded ids; drop self-reference
    $self_num = $name.Substring(0, 2)
    $actual_set = $refs | ForEach-Object { "0$_" } |
                  Where-Object { $_ -ne $self_num } | Sort-Object -Unique
    $expected_set = $ExpectedRefs[$name] | Sort-Object -Unique

    $refs_actual = $actual_set -join ","
    $refs_expected = $expected_set -join ","

    # delta: expected refs missing from actual
    $missing = $expected_set | Where-Object { $_ -notin $actual_set }
    $delta = ($missing | Sort-Object -Unique) -join ","

    $Results += [PSCustomObject]@{
        spec = $name
        refs_actual = $refs_actual
        refs_expected = $refs_expected
        delta = $delta
    }
}

# Write ref_matrix.csv
$Results | Export-Csv -Path "ref_matrix.csv" -NoTypeInformation -Encoding UTF8

# Terminal summary
Write-Host "`n=== Pkg-07 SDD Reference Matrix ===" -ForegroundColor Cyan
$Results | Format-Table -AutoSize

# Acceptance: delta column must be all empty
$any_delta = $Results | Where-Object { $_.delta -ne "" }
if ($any_delta) {
    Write-Host "`n[FAIL] Spec refs missing (delta non-empty):" -ForegroundColor Red
    $any_delta | Format-Table -AutoSize
    exit 1
} else {
    Write-Host "`n[PASS] All spec refs compliant (delta all empty)" -ForegroundColor Green
    Write-Host "ref_matrix.csv generated (UTF-8)" -ForegroundColor Green
}
