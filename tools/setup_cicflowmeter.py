#!/usr/bin/env python3
"""
Script tự động tải và setup CICFlowMeter
CICFlowMeter là tool Java để extract 79 CICIDS2017 features từ pcap files
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Màu sắc cho output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def check_java():
    """Kiểm tra Java có được cài đặt không"""
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stderr.split('\n')[0] if result.stderr else result.stdout.split('\n')[0]
            print_success(f"Java found: {version}")
            return True
    except:
        pass
    
    print_error("Java not found!")
    print_info("Please install Java JDK 8 or higher:")
    print_info("  Windows: Download from https://www.oracle.com/java/technologies/downloads/")
    print_info("  Linux: sudo apt-get install openjdk-11-jdk")
    print_info("  macOS: brew install openjdk")
    return False

def check_maven():
    """Kiểm tra Maven có được cài đặt không"""
    try:
        result = subprocess.run(['mvn', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_success(f"Maven found: {version}")
            return True
    except:
        pass
    
    print_warning("Maven not found - will try to download pre-built JAR")
    print_info("If you want to build from source, install Maven:")
    print_info("  Windows: Download from https://maven.apache.org/download.cgi")
    print_info("  Linux: sudo apt-get install maven")
    print_info("  macOS: brew install maven")
    return False

def download_file(url, dest_path):
    """Download file từ URL"""
    try:
        print_info(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print_success(f"Downloaded to {dest_path}")
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False

def clone_and_build_cicflowmeter(tools_dir):
    """Clone CICFlowMeter từ GitHub và build"""
    cicflowmeter_dir = tools_dir / 'CICFlowMeter'
    
    # Clone repository
    if not cicflowmeter_dir.exists():
        print_info("Cloning CICFlowMeter from GitHub...")
        try:
            subprocess.run(
                ['git', 'clone', 'https://github.com/ahlashkari/CICFlowMeter.git', str(cicflowmeter_dir)],
                check=True,
                timeout=300
            )
            print_success("Repository cloned")
        except subprocess.TimeoutExpired:
            print_error("Clone timeout - repository may be too large")
            return False
        except subprocess.CalledProcessError as e:
            print_error(f"Git clone failed: {e}")
            return False
        except FileNotFoundError:
            print_error("Git not found! Please install Git or download manually")
            return False
    else:
        print_info("CICFlowMeter directory already exists")
    
    # Build với Maven
    if check_maven():
        print_info("Building CICFlowMeter with Maven...")
        try:
            os.chdir(cicflowmeter_dir)
            result = subprocess.run(
                ['mvn', 'clean', 'install', '-DskipTests'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            if result.returncode == 0:
                # Tìm JAR file trong target/
                jar_files = list((cicflowmeter_dir / 'target').glob('*.jar'))
                if jar_files:
                    # Copy JAR ra ngoài để dễ truy cập
                    main_jar = jar_files[0]
                    final_jar = cicflowmeter_dir / 'CICFlowMeter.jar'
                    shutil.copy2(main_jar, final_jar)
                    print_success(f"Build successful! JAR file: {final_jar}")
                    return True
                else:
                    print_warning("Build completed but JAR file not found")
            else:
                print_error(f"Build failed: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print_error("Build timeout")
        except Exception as e:
            print_error(f"Build error: {e}")
        finally:
            os.chdir(Path.cwd())
    
    return False

def find_prebuilt_jar():
    """Tìm pre-built JAR file từ releases"""
    print_info("Looking for pre-built JAR file...")
    
    # Các URL có thể có pre-built JAR (cần cập nhật theo thực tế)
    possible_urls = [
        # Có thể thêm các URL release từ GitHub
        # "https://github.com/ahlashkari/CICFlowMeter/releases/download/v4.0/CICFlowMeter.jar"
    ]
    
    # Hoặc hướng dẫn download manual
    print_warning("Pre-built JAR not available automatically")
    print_info("Please download manually from:")
    print_info("  https://github.com/ahlashkari/CICFlowMeter/releases")
    print_info("Or build from source using Maven")
    
    return False

def setup_cicflowmeter():
    """Setup CICFlowMeter"""
    print("=" * 60)
    print("CICFlowMeter Setup Script")
    print("=" * 60)
    
    # Tạo thư mục tools
    project_root = Path(__file__).parent.parent
    tools_dir = project_root / 'tools'
    tools_dir.mkdir(exist_ok=True)
    
    print_info(f"Tools directory: {tools_dir}")
    
    # Kiểm tra Java
    if not check_java():
        return False
    
    cicflowmeter_dir = tools_dir / 'CICFlowMeter'
    cicflowmeter_jar = cicflowmeter_dir / 'CICFlowMeter.jar'
    
    # Kiểm tra xem đã có chưa
    if cicflowmeter_jar.exists():
        print_success(f"CICFlowMeter already exists at: {cicflowmeter_jar}")
        return True
    
    # Thử build từ source
    print_info("Attempting to build from source...")
    if clone_and_build_cicflowmeter(tools_dir):
        return True
    
    # Nếu build thất bại, hướng dẫn manual
    print_warning("Automatic setup failed")
    print_info("=" * 60)
    print_info("Manual Setup Instructions:")
    print_info("=" * 60)
    print_info("1. Clone repository:")
    print_info(f"   git clone https://github.com/ahlashkari/CICFlowMeter.git {cicflowmeter_dir}")
    print_info("")
    print_info("2. Build with Maven:")
    print_info(f"   cd {cicflowmeter_dir}")
    print_info("   mvn clean install -DskipTests")
    print_info("")
    print_info("3. Copy JAR file:")
    print_info(f"   Copy target/*.jar to {cicflowmeter_jar}")
    print_info("")
    print_info("Or download pre-built JAR from:")
    print_info("   https://github.com/ahlashkari/CICFlowMeter/releases")
    print_info(f"   Place it at: {cicflowmeter_jar}")
    
    return False

if __name__ == '__main__':
    success = setup_cicflowmeter()
    sys.exit(0 if success else 1)

