# Define the test name parameter
param (
    [string]$TestName = "YouTubeTopic"
)

# --- Configuration ---
$ApiBaseUrl = "http://127.0.0.1:8000/api/v1"
$Headers = @{ "Content-Type" = "application/json" }
$requestBody = ""

# --- Test Definitions ---
if ($TestName -eq "YouTubeTopic") {
    Write-Host "--- Running Test: YouTube Topic Search ---"
    $requestBody = @'
{
    "query": "Machine Learning Explained",
    "search_type": "topic",
    "filters": {
        "platforms": ["YouTube"],
        "min_followers": 100000
    },
    "limit": 3
}
'@
}
elseif ($TestName -eq "YouTubeCreator") {
    Write-Host "--- Running Test: YouTube Creator Search ---"
    $requestBody = @'
{
    "query": "Marques Brownlee",
    "search_type": "creator",
    "filters": {
        "platforms": ["YouTube"]
    },
    "limit": 1
}
'@
}
elseif ($TestName -eq "InstagramCreator") {
    Write-Host "--- Running Test: Instagram Creator Search ---"
    $requestBody = @'
{
    "query": "mkbhd",
    "search_type": "creator",
    "filters": {
        "platforms": ["Instagram"]
    }
}
'@
}
else {
    Write-Error "Invalid test name provided: '$TestName'"
    exit 1
}

# --- Main Execution ---
Write-Host "--- Request Body ---"
Write-Host $requestBody
Write-Host "--------------------"

try {
    Write-Host "Calling API..."
    $response = Invoke-RestMethod -Method Post -Uri "$ApiBaseUrl/search" -Headers $Headers -Body $requestBody -ErrorAction Stop
    
    Write-Host "--- API Response (Success) ---"
    Write-Host ($response | ConvertTo-Json -Depth 10)
    Write-Host "----------------------------"
}
catch {
    Write-Host "--- API Call FAILED ---" -ForegroundColor Red
    $errorDetails = $_.Exception
    Write-Host "Error Type: $($errorDetails.GetType().FullName)"
    
    if ($errorDetails.Response) {
        $statusCode = [int]$errorDetails.Response.StatusCode
        Write-Host "Status Code: $statusCode"
        
        $stream = $errorDetails.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $errorBody = $reader.ReadToEnd()
        $reader.Close()
        $stream.Close()
        
        Write-Host "Error Body: $errorBody"
    } else {
        Write-Host "Error Message: $($errorDetails.Message)"
    }
    Write-Host "-----------------------"
}

Write-Host "--- Test Finished ---"