# Airflow Setup Script
# Run this to initialize Airflow and create your first admin user

Write-Host "=" -ForegroundColor Cyan
Write-Host "Airflow Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Set Airflow home directory
$AIRFLOW_HOME = Join-Path $PSScriptRoot "."
$env:AIRFLOW_HOME = $AIRFLOW_HOME

Write-Host "✓ Airflow home: $AIRFLOW_HOME" -ForegroundColor Green
Write-Host ""

# 2. Initialize database
Write-Host "Initializing Airflow database..." -ForegroundColor Yellow
uv run airflow db migrate

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Database initialized" -ForegroundColor Green
} else {
    Write-Host "✗ Database initialization failed" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 3. Create admin user
Write-Host "Creating admin user..." -ForegroundColor Yellow
uv run airflow users create `
    --username admin `
    --firstname Admin `
    --lastname User `
    --role Admin `
    --email admin@example.com `
    --password admin

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Admin user created" -ForegroundColor Green
} else {
    Write-Host "Note: User might already exist" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Start Airflow:" -ForegroundColor White
Write-Host "     uv run airflow standalone" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Open browser:" -ForegroundColor White
Write-Host "     http://localhost:8080" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Login with:" -ForegroundColor White
Write-Host "     Username: admin" -ForegroundColor Cyan
Write-Host "     Password: admin" -ForegroundColor Cyan
Write-Host ""
Write-Host "  4. Start learning with tutorial DAGs:" -ForegroundColor White
Write-Host "     - 01_hello_airflow" -ForegroundColor Cyan
Write-Host "     - 02_crash_ml_pipeline" -ForegroundColor Cyan
Write-Host "     - 03_advanced_concepts" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tip: Press Ctrl+C to stop Airflow" -ForegroundColor Yellow
Write-Host ""
