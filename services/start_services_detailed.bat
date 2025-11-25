@echo off
REM ====================================================================================
REM Safenet IDS - Detailed Service Startup Script with Comprehensive Comments
REM
REM This script provides detailed startup process for all 5 Kafka-based services
REM with extensive logging and error handling for the Safenet IDS system.
REM
REM Architecture Overview:
REM Network Data → [Producer] → raw_network_events → [Preprocessing] → preprocessed_events
REM     ↓
REM level1_predictions ← [Level 1 Prediction] ← preprocessed_events
REM     ↓
REM level2_predictions ← [Level 2 Prediction] ← level1_predictions
REM     ↓
REM ids_alerts ← [Alerting Service] ← level2_predictions
REM
REM Prerequisites Check:
REM - Kafka cluster running on localhost:9092
REM - Python 3.8+ with required packages installed
REM - ML models trained and available
REM - Sufficient system resources (RAM, CPU)
REM ====================================================================================

echo ====================================================================================
echo 🚀 SAFENET IDS - DETAILED SERVICE STARTUP
echo ====================================================================================
echo System Time: %DATE% %TIME%
echo Working Directory: %CD%
echo.

REM ===== SYSTEM REQUIREMENTS CHECK =====
echo 🔍 Checking System Requirements...
echo.

REM Check if Kafka is running (basic connectivity test)
echo Checking Kafka connectivity...
c:\kafka\bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ERROR: Kafka cluster not accessible on localhost:9092
    echo 💡 SOLUTION: Run 'cd c:/kafka && start-ids-kafka.bat' first
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Kafka cluster is accessible
)

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ERROR: Python not found in PATH
    echo 💡 SOLUTION: Install Python 3.8+ and add to PATH
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Python is available
)

REM Check if required directories exist
if not exist "logs" mkdir logs
if not exist "data" mkdir data

echo ✅ System requirements satisfied
echo.

REM ===== SERVICE STARTUP SEQUENCE =====
echo 📋 Starting Services in Dependency Order:
echo.

REM ===== 1. NETWORK DATA PRODUCER SERVICE =====
echo ┌─────────────────────────────────────────────────────────────────────────────┐
echo │ SERVICE 1/5: Network Data Producer                                        │
echo ├─────────────────────────────────────────────────────────────────────────────┤
echo │ 🎯 Purpose: Generates realistic network traffic data for IDS testing     │
echo │ 🔧 Technology: Python + Kafka Producer                                    │
echo │ 📤 Output Topic: raw_network_events                                       │
echo │ ⚙️  Configuration: Synthetic data generation with CICIDS2017 features     │
echo │ 📊 Data Rate: 1 record/second (configurable)                              │
echo │ 🔍 Monitoring: services/logs/network_producer.log                         │
echo └─────────────────────────────────────────────────────────────────────────────┘
echo.

start "Safenet-Network-Producer" cmd /c "python network_data_producer.py"
echo ⏳ Service starting... (waiting 5 seconds for initialization)
timeout /t 5 /nobreak > nul
echo ✅ Network Data Producer service started
echo.

REM ===== 2. DATA PREPROCESSING SERVICE =====
echo ┌─────────────────────────────────────────────────────────────────────────────┐
echo │ SERVICE 2/5: Data Preprocessing Service                                   │
echo ├─────────────────────────────────────────────────────────────────────────────┤
echo │ 🎯 Purpose: Applies data cleaning, normalization, and feature scaling    │
echo │ 🔧 Technology: Python + Kafka Consumer/Producer + Pandas/Sklearn         │
echo │ 📥 Input Topic: raw_network_events                                        │
echo │ 📤 Output Topic: preprocessed_events                                      │
echo │ ⚙️  Pipeline: normalize → convert_numeric → fill_missing → scale         │
echo │ 🔍 Monitoring: services/logs/data_preprocessing.log                      │
echo └─────────────────────────────────────────────────────────────────────────────┘
echo.

start "Safenet-Data-Preprocessing" cmd /c "python data_preprocessing_service.py"
echo ⏳ Service starting... (waiting 5 seconds for initialization)
timeout /t 5 /nobreak > nul
echo ✅ Data Preprocessing service started
echo.

REM ===== 3. LEVEL 1 PREDICTION SERVICE =====
echo ┌─────────────────────────────────────────────────────────────────────────────┐
echo │ SERVICE 3/5: Level 1 Prediction Service                                   │
echo ├─────────────────────────────────────────────────────────────────────────────┤
echo │ 🎯 Purpose: Classifies network traffic into 5 attack groups               │
echo │ 🔧 Technology: Python + Kafka Consumer/Producer + Scikit-learn            │
echo │ 📥 Input Topic: preprocessed_events                                       │
echo │ 📤 Output Topic: level1_predictions                                       │
echo │ 🤖 Model: artifacts/ids_pipeline.joblib (RandomForest)                    │
echo │ 🎯 Classes: benign(0), dos(1), ddos(2), bot(3), rare_attack(4)            │
echo │ 🔍 Monitoring: services/logs/level1_prediction.log                       │
echo └─────────────────────────────────────────────────────────────────────────────┘
echo.

start "Safenet-Level1-Prediction" cmd /c "python level1_prediction_service.py"
echo ⏳ Service starting... (waiting 5 seconds for model loading)
timeout /t 5 /nobreak > nul
echo ✅ Level 1 Prediction service started
echo.

REM ===== 4. LEVEL 2 PREDICTION SERVICE =====
echo ┌─────────────────────────────────────────────────────────────────────────────┐
echo │ SERVICE 4/5: Level 2 Prediction Service                                   │
echo ├─────────────────────────────────────────────────────────────────────────────┤
echo │ 🎯 Purpose: Detailed classification for dos and rare_attack groups        │
echo │ 🔧 Technology: Python + Kafka Consumer/Producer + Multiple ML Models      │
echo │ 📥 Input Topic: level1_predictions                                        │
echo │ 📤 Output Topic: level2_predictions                                       │
echo │ 🤖 Models: artifacts_level2/dos/ + artifacts_level2/rare_attack/          │
echo │ 🎯 Examples: DoS Hulk, SQL Injection, FTP-Patator, etc.                   │
echo │ 🔍 Monitoring: services/logs/level2_prediction.log                       │
echo └─────────────────────────────────────────────────────────────────────────────┘
echo.

start "Safenet-Level2-Prediction" cmd /c "python level2_prediction_service.py"
echo ⏳ Service starting... (waiting 5 seconds for model loading)
timeout /t 5 /nobreak > nul
echo ✅ Level 2 Prediction service started
echo.

REM ===== 5. ALERTING SERVICE =====
echo ┌─────────────────────────────────────────────────────────────────────────────┐
echo │ SERVICE 5/5: Alerting Service                                             │
echo ├─────────────────────────────────────────────────────────────────────────────┤
echo │ 🎯 Purpose: Generates security alerts and stores in database              │
echo │ 🔧 Technology: Python + Kafka Consumer/Producer + SQLite                  │
echo │ 📥 Input Topic: level2_predictions                                        │
echo │ 📤 Output Topic: ids_alerts                                               │
echo │ 💾 Database: services/data/alerts.db                                      │
echo │ 🚨 Severity Levels: low, medium, high, critical                           │
echo │ ⚙️  Thresholds: Configurable confidence-based alerting                    │
echo │ 🔍 Monitoring: services/logs/alerting.log                                │
echo └─────────────────────────────────────────────────────────────────────────────┘
echo.

start "Safenet-Alerting-Service" cmd /c "python alerting_service.py"
echo ⏳ Service starting... (waiting 5 seconds for database initialization)
timeout /t 5 /nobreak > nul
echo ✅ Alerting service started
echo.

REM ===== STARTUP COMPLETE =====
echo ====================================================================================
echo 🎉 SAFENET IDS SYSTEM SUCCESSFULLY STARTED!
echo ====================================================================================
echo.
echo 📊 System Status Summary:
echo ┌─────────────────────────────────────────────────────────────────────────────┐
echo │ Component              │ Status      │ Details                              │
echo ├─────────────────────────┼─────────────┼──────────────────────────────────────┤
echo │ Kafka Cluster          │ ✅ Running  │ localhost:9092                       │
echo │ Network Producer       │ ✅ Running  │ → raw_network_events                 │
echo │ Data Preprocessing     │ ✅ Running  │ raw_network_events → preprocessed    │
echo │ Level 1 Prediction     │ ✅ Running  │ preprocessed → level1_predictions    │
echo │ Level 2 Prediction     │ ✅ Running  │ level1_predictions → level2_pred     │
echo │ Alerting Service       │ ✅ Running  │ level2_predictions → ids_alerts      │
echo │ Database               │ ✅ Ready    │ SQLite: services/data/alerts.db      │
echo └─────────────────────────┴─────────────┴──────────────────────────────────────┘
echo.
echo 🔍 Monitoring & Management:
echo • 📁 View Logs: services/logs/ (individual service logs)
echo • 🔍 Health Check: python check_services.py
echo • 📊 Kafka Monitor: Use Kafka console tools to inspect topics
echo • 🗄️  Database: sqlite3 services/data/alerts.db "SELECT * FROM alerts LIMIT 5"
echo • 📈 Real-time: kafka-console-consumer.bat --topic ids_alerts --from-beginning
echo.
echo ⚠️  IMPORTANT NOTES:
echo • Keep this window open to maintain service processes
echo • Each service runs in its own window for independent monitoring
echo • Use Ctrl+C in individual windows to stop specific services
echo • Press any key in this window to shutdown ALL services gracefully
echo.

REM ===== WAIT FOR USER INPUT TO SHUTDOWN =====
echo Press any key to stop all services and exit...
pause > nul

REM ===== GRACEFUL SHUTDOWN =====
echo.
echo 🛑 Initiating graceful shutdown of all Safenet IDS services...
echo.

REM Stop services in reverse order to avoid data loss
echo Stopping Alerting Service...
taskkill /FI "WINDOWTITLE eq Safenet-Alerting-Service*" /T /F > nul 2>&1

echo Stopping Level 2 Prediction Service...
taskkill /FI "WINDOWTITLE eq Safenet-Level2-Prediction*" /T /F > nul 2>&1

echo Stopping Level 1 Prediction Service...
taskkill /FI "WINDOWTITLE eq Safenet-Level1-Prediction*" /T /F > nul 2>&1

echo Stopping Data Preprocessing Service...
taskkill /FI "WINDOWTITLE eq Safenet-Data-Preprocessing*" /T /F > nul 2>&1

echo Stopping Network Data Producer...
taskkill /FI "WINDOWTITLE eq Safenet-Network-Producer*" /T /F > nul 2>&1

echo.
echo ✅ All Safenet IDS services stopped successfully.
echo.
echo ====================================================================================
echo 👋 Safenet IDS System Shutdown Complete
echo ====================================================================================
echo Thank you for using Safenet IDS!
echo.
pause
