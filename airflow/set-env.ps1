# Airflow environment configuration
# This file is read by Airflow on startup

# Set the Airflow home directory
# This should match the directory containing this file
$env:AIRFLOW_HOME = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Airflow home set to: $env:AIRFLOW_HOME" -ForegroundColor Green
