@echo off
echo ============================================
echo    Safenet IDS - Starting All Services
echo ============================================
echo.

echo Starting services in order...
echo.

echo 1. Starting Network Data Producer Service...
start "Network Producer" cmd /c "python network_data_producer.py"

echo Waiting 5 seconds for producer to start...
timeout /t 5 /nobreak > nul

echo 2. Starting Data Preprocessing Service...
start "Data Preprocessing" cmd /c "python data_preprocessing_service.py"

echo Waiting 5 seconds for preprocessing to start...
timeout /t 5 /nobreak > nul

echo 3. Starting Level 1 Prediction Service...
start "Level 1 Prediction" cmd /c "python level1_prediction_service.py"

echo Waiting 5 seconds for Level 1 to start...
timeout /t 5 /nobreak > nul

echo 4. Starting Level 2 Prediction Service...
start "Level 2 Prediction" cmd /c "python level2_prediction_service.py"

echo Waiting 5 seconds for Level 2 to start...
timeout /t 5 /nobreak > nul

echo 5. Starting Alerting Service...
start "Alerting Service" cmd /c "python alerting_service.py"

echo.
echo ============================================
echo    All Safenet IDS Services Started!
echo ============================================
echo.
echo Services running:
echo - Network Data Producer (raw_network_events)
echo - Data Preprocessing (preprocessed_events)
echo - Level 1 Prediction (level1_predictions)
echo - Level 2 Prediction (level2_predictions)
echo - Alerting Service (ids_alerts)
echo.
echo Check logs in services/logs/ for monitoring
echo.
echo Press any key to stop all services...
pause > nul

echo Stopping all services...
taskkill /FI "WINDOWTITLE eq Network Producer*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq Data Preprocessing*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq Level 1 Prediction*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq Level 2 Prediction*" /T /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq Alerting Service*" /T /F > nul 2>&1

echo All services stopped.
pause
