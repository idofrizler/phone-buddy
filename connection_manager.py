"""Connection Manager for wireless ADB connections."""

import subprocess
import time
from typing import Tuple
import uiautomator2 as u2


class ConnectionManager:
    """Manages ADB connections to Android devices (USB or wireless)."""
    
    def __init__(self, device_ip: str = None, port: int = 5555, use_usb: bool = False):
        self.device_ip = device_ip
        self.port = port
        self.use_usb = use_usb
        self.device_address = None if use_usb else f"{device_ip}:{port}"
        self.device = None
    
    def _run_adb(self, args: list, timeout: int = 30) -> Tuple[bool, str]:
        """Run an ADB command and return success status and output."""
        try:
            result = subprocess.run(
                ["adb"] + args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, "ADB not found. Please install Android SDK Platform Tools."
    
    def setup_tcpip(self) -> bool:
        """Setup ADB TCP/IP mode on the device (requires USB connection first)."""
        print(f"Setting up TCP/IP mode on port {self.port}...")
        success, output = self._run_adb(["tcpip", str(self.port)])
        if success:
            print("TCP/IP mode enabled. You can now disconnect USB.")
            time.sleep(2)
        else:
            print(f"Failed to enable TCP/IP: {output}")
        return success
    
    def connect(self) -> bool:
        """Connect to the device (USB or wireless)."""
        if self.use_usb:
            return self._connect_usb()
        else:
            return self._connect_wireless()
    
    def _connect_usb(self) -> bool:
        """Connect via USB."""
        print("Connecting via USB...")
        success, output = self._run_adb(["devices"])
        if "device" in output and "unauthorized" not in output.lower():
            # Get the device serial
            lines = output.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if '\tdevice' in line:
                    self.device_address = line.split('\t')[0]
                    print(f"✓ Found USB device: {self.device_address}")
                    return self._init_uiautomator()
        print(f"✗ No USB device found: {output}")
        return False
    
    def _connect_wireless(self) -> bool:
        """Connect wirelessly."""
        print(f"Connecting to {self.device_address}...")
        
        # First try to disconnect any existing connection
        self._run_adb(["disconnect", self.device_address])
        
        # Attempt connection
        success, output = self._run_adb(["connect", self.device_address])
        
        if "connected" in output.lower() and "cannot" not in output.lower():
            print(f"✓ Connected to {self.device_address}")
            return self._init_uiautomator()
        else:
            print(f"✗ Failed to connect: {output}")
            return False
    
    def _init_uiautomator(self) -> bool:
        """Initialize uiautomator2 connection."""
        try:
            self.device = u2.connect(self.device_address)
            info = self.device.info
            print(f"✓ UIAutomator2 connected: {info.get('productName', 'Unknown Device')}")
            return True
        except Exception as e:
            print(f"✗ UIAutomator2 connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the device."""
        if not self.use_usb and self.device_address:
            self._run_adb(["disconnect", self.device_address])
        self.device = None
        print("Disconnected")
    
    def is_connected(self) -> bool:
        """Check if device is connected."""
        if self.device is None:
            return False
        try:
            self.device.info
            return True
        except:
            return False
    
    def get_device(self):
        """Get the uiautomator2 device object."""
        return self.device
