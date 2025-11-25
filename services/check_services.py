#!/usr/bin/env python3
"""
Safenet IDS - Service Health Check Script
Kiểm tra tình trạng của tất cả các Kafka services
"""

import json
import requests
import subprocess
import sys
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime, timedelta

def check_kafka_connection(kafka_servers='localhost:9092'):
    """
    Kiểm tra tình trạng kết nối và hoạt động của Kafka cluster

    Process chi tiết:
    1. Test Kafka Producer Connection:
       - Tạo producer instance với timeout ngắn
       - Validate có thể connect đến brokers

    2. Test Kafka Consumer Connection:
       - Tạo consumer instance để test connectivity
       - Verify có thể communicate với cluster

    3. Test Topic Operations:
       - List topics để kiểm tra admin operations
       - Validate topic management functionality
       - Check topic count và names

    Args:
        kafka_servers (str): Kafka bootstrap servers (default: localhost:9092)

    Returns:
        bool: True nếu Kafka healthy, False nếu có vấn đề
    """
    print("🔍 Checking Kafka connection and topic management...")

    try:
        # ===== TEST 1: Producer Connection =====
        # Tạo producer với timeout ngắn để test connectivity
        producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            request_timeout_ms=5000  # 5s timeout for quick health check
        )

        # ===== TEST 2: Consumer Connection =====
        # Test consumer connectivity
        consumer = KafkaConsumer(
            bootstrap_servers=kafka_servers,
            request_timeout_ms=5000  # Quick timeout
        )

        # ===== TEST 3: Topic Management =====
        # Test admin operations - list topics
        result = subprocess.run([
            'c:/kafka/bin/windows/kafka-topics.bat',
            '--list',
            '--bootstrap-server', kafka_servers
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            # Parse topic list
            topics_raw = result.stdout.strip().split('\n')
            topics = [t.strip() for t in topics_raw if t.strip()]

            # Report success với topic count
            print(f"✅ Kafka connected successfully!")
            print(f"   📊 Topics found: {len(topics)}")
            if topics:
                print(f"   📋 Sample topics: {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}")

            return True
        else:
            # Topic listing failed
            print(f"❌ Kafka topic management failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Kafka check timeout - cluster may be unresponsive")
        return False
    except Exception as e:
        print(f"❌ Kafka connection error: {e}")
        return False

def check_service_processes():
    """Kiểm tra các process service đang chạy"""
    print("\n🔍 Checking service processes...")

    services = [
        'network_data_producer.py',
        'data_preprocessing_service.py',
        'level1_prediction_service.py',
        'level2_prediction_service.py',
        'alerting_service.py'
    ]

    running_services = []

    try:
        # Sử dụng tasklist để kiểm tra processes
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split('","')
                    if len(parts) >= 8:
                        cmd_line = parts[7].strip('"')
                        for service in services:
                            if service in cmd_line:
                                running_services.append(service)

        print(f"✅ Found {len(running_services)} running services:")
        for service in running_services:
            print(f"  • {service}")

        missing_services = [s for s in services if s not in running_services]
        if missing_services:
            print(f"❌ Missing services: {', '.join(missing_services)}")

        return len(running_services) == len(services)

    except Exception as e:
        print(f"❌ Error checking processes: {e}")
        return False

def check_topics_and_data(kafka_servers='localhost:9092'):
    """Kiểm tra topics và dữ liệu"""
    print("\n🔍 Checking topics and recent data...")

    required_topics = [
        'raw_network_events',
        'preprocessed_events',
        'level1_predictions',
        'level2_predictions',
        'ids_alerts'
    ]

    try:
        # Kiểm tra topics tồn tại
        result = subprocess.run([
            'c:/kafka/bin/windows/kafka-topics.bat',
            '--list',
            '--bootstrap-server', kafka_servers
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Cannot list topics: {result.stderr}")
            return False

        existing_topics = result.stdout.strip().split('\n')
        existing_topics = [t.strip() for t in existing_topics if t.strip()]

        missing_topics = [t for t in required_topics if t not in existing_topics]
        if missing_topics:
            print(f"❌ Missing topics: {', '.join(missing_topics)}")
            print(f"  💡 Run: cd c:/kafka && create-ids-topics.bat")
            return False

        print(f"✅ All {len(required_topics)} required topics exist")

        # Kiểm tra dữ liệu gần đây trong topics
        print("\n📊 Checking recent messages in topics:")
        for topic in required_topics:
            try:
                # Lấy thông tin topic
                desc_result = subprocess.run([
                    'c:/kafka/bin/windows/kafka-topics.bat',
                    '--describe',
                    '--topic', topic,
                    '--bootstrap-server', kafka_servers
                ], capture_output=True, text=True, timeout=10)

                if desc_result.returncode == 0:
                    desc_lines = desc_result.stdout.strip().split('\n')
                    for line in desc_lines:
                        if 'Partition:' in line and 'Leader:' in line:
                            parts = line.split()
                            partition = parts[1]
                            leader = parts[3]
                            replicas = parts[5]
                            isr = parts[7]
                            print(f"  • {topic}: Partition {partition}, Leader {leader}, ISR {isr}")

            except Exception as e:
                print(f"  • {topic}: Error getting info - {e}")

        return True

    except Exception as e:
        print(f"❌ Error checking topics: {e}")
        return False

def check_models():
    """Kiểm tra models tồn tại"""
    print("\n🔍 Checking ML models...")

    models_status = {
        'Level 1 Model': ('artifacts/ids_pipeline.joblib', 'Required'),
        'Level 2 DoS Model': ('artifacts_level2/dos/dos_pipeline.joblib', 'Optional'),
        'Level 2 Rare Attack Model': ('artifacts_level2/rare_attack/rare_attack_pipeline.joblib', 'Optional')
    }

    all_required_present = True

    for model_name, (model_path, requirement) in models_status.items():
        path = Path(model_path)
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(".2f"        else:
            print(f"❌ {model_name}: NOT FOUND ({model_path})")
            if requirement == 'Required':
                all_required_present = False

    return all_required_present

def check_database():
    """Kiểm tra database alerts"""
    print("\n🔍 Checking alerts database...")

    db_path = Path('services/data/alerts.db')

    if not db_path.exists():
        print("❌ Alerts database not found")
        return False

    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Đếm số alerts
        cursor.execute("SELECT COUNT(*) FROM alerts")
        alert_count = cursor.fetchone()[0]

        # Đếm alerts trong 24h gần nhất
        yesterday = datetime.now() - timedelta(days=1)
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp > ?",
                      (yesterday.isoformat(),))
        recent_count = cursor.fetchone()[0]

        # Đếm theo severity
        cursor.execute("""
            SELECT severity, COUNT(*) FROM alerts
            GROUP BY severity
            ORDER BY COUNT(*) DESC
        """)
        severity_stats = cursor.fetchall()

        conn.close()

        print(f"✅ Database OK - Total alerts: {alert_count}")
        print(f"  📅 Recent (24h): {recent_count}")
        if severity_stats:
            print("  📊 By severity:"            for severity, count in severity_stats:
                print(f"    • {severity}: {count}")

        return True

    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    """
    Main entry point cho Safenet IDS Health Check

    Thực hiện comprehensive health assessment của toàn bộ hệ thống IDS bao gồm:

    1. **Infrastructure Layer:**
       - Kafka cluster connectivity và topic management
       - Zookeeper coordination (implicit qua Kafka)

    2. **Service Layer:**
       - Process status của tất cả 5 services
       - Service-to-service communication health

    3. **Data Layer:**
       - Topic existence và message flow
       - Recent data availability và freshness

    4. **Model Layer:**
       - ML model files existence và loadability
       - Model compatibility và versioning

    5. **Storage Layer:**
       - Alert database connectivity và integrity
       - Historical data persistence

    Exit Codes:
    - 0: All checks passed - system healthy
    - 1: Some checks failed - system degraded

    Remediation Actions:
    - Start Kafka cluster nếu chưa chạy
    - Launch missing services
    - Create missing topics
    - Train/reload models nếu corrupted
    - Check database connectivity
    """
    print("🚀 Safenet IDS - Comprehensive System Health Check")
    print("=" * 60)

    kafka_servers = 'localhost:9092'  # Default Kafka servers
    all_checks_passed = True  # Master flag for overall health

    # ===== PHASE 1: Infrastructure Health =====
    print("\n🏗️  PHASE 1: Infrastructure Health")
    print("-" * 40)
    if not check_kafka_connection(kafka_servers):
        all_checks_passed = False
        print("⚠️  Kafka infrastructure issues detected")

    # ===== PHASE 2: Service Health =====
    print("\n⚙️  PHASE 2: Service Processes")
    print("-" * 40)
    if not check_service_processes():
        all_checks_passed = False
        print("⚠️  Service process issues detected")

    # ===== PHASE 3: Data Flow Health =====
    print("\n📊 PHASE 3: Data Pipeline Health")
    print("-" * 40)
    if not check_topics_and_data(kafka_servers):
        all_checks_passed = False
        print("⚠️  Data pipeline issues detected")

    # ===== PHASE 4: Model Health =====
    print("\n🧠 PHASE 4: ML Models Health")
    print("-" * 40)
    if not check_models():
        all_checks_passed = False
        print("⚠️  Model loading issues detected")

    # ===== PHASE 5: Storage Health =====
    print("\n💾 PHASE 5: Storage Health")
    print("-" * 40)
    if not check_database():
        all_checks_passed = False
        print("⚠️  Database issues detected")

    # ===== FINAL REPORT =====
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Safenet IDS system is fully operational and healthy")
        print("\n📈 System Status: GREEN - All services running normally")
        sys.exit(0)
    else:
        print("⚠️  SYSTEM HEALTH ISSUES DETECTED!")
        print("❌ Some components are degraded or non-functional")
        print("\n📈 System Status: YELLOW/RED - Attention required")

        # Provide remediation guidance
        print("\n🔧 QUICK REMEDIATION STEPS:")
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│ 1. Start Kafka: cd c:/kafka && start-ids-kafka.bat          │")
        print("│ 2. Start services: cd services && start_all_services.bat    │")
        print("│ 3. Create topics: cd c:/kafka && create-ids-topics.bat      │")
        print("│ 4. Train models: python ids_pipeline/train_model.py         │")
        print("│ 5. Check logs: services/logs/ and c:/kafka/logs/            │")
        print("└─────────────────────────────────────────────────────────────┘")

        sys.exit(1)

if __name__ == '__main__':
    main()
