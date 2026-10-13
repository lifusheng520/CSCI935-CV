[CmdletBinding()]
param(
    [string]$ApprovedConfig,
    [string]$QcApproval,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Project Python missing' }
if ($Execute -and (-not $ApprovedConfig -or -not $QcApproval)) {
    throw 'Execute requires independent formal data and frozen schema-v1 approved config/QC policy. STEP 6 does not authorize execution.'
}
Push-Location -LiteralPath $project
try {
    if (-not $ApprovedConfig -and -not $QcApproval) {
        & $python -B -m src.member5_mdb_runner plan
    } else {
        if (-not $ApprovedConfig -or -not $QcApproval) { throw 'Both approved configuration and QC policy are required' }
        $runnerArgs = @('-B','-m','src.member5_mdb_runner','run','--approved-config',$ApprovedConfig,'--qc-approval',$QcApproval,'--mode','formal')
        if ($Execute) { $runnerArgs += '--execute' }
        & $python @runnerArgs
    }
    $runnerExit = $LASTEXITCODE
} finally { Pop-Location }
exit $runnerExit
