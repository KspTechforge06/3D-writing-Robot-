"""
Hardware Interface for Writing Robot

Wraps serial communication with Arduino/HW-130 shield.
Provides both synchronous and async interfaces.
"""
import serial
import serial.tools.list_ports
import threading
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotorPosition:
    x: float
    y: float
    timestamp: float


class HardwareInterface:
    """
    Communicates with Arduino running dvd_stepper_serial.ino firmware.
    
    Protocol (9600 baud):
    - x <steps>   : Move X motor
    - y <steps>   : Move Y motor
    - b <steps>   : Move both
    - r           : Return to home (0,0)
    - s <rpm>     : Set speed
    - p <angle>   : Set pen servo angle (if implemented)
    """
    
    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 9600,
        timeout: float = 2.0,
        steps_per_mm: float = 5.0,  # default calibration (X: 5.0, Y: 4.8 for this robot)
        steps_per_mm_x: Optional[float] = None,  # per-axis calibration override
        steps_per_mm_y: Optional[float] = None,
        max_steps: Tuple[int, int] = (275, 250),  # measured dead ends (0,0 corner)
        safety_margin_steps: int = 15,  # never drive into the end stops
    ):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.steps_per_mm_x = steps_per_mm_x if steps_per_mm_x is not None else steps_per_mm
        self.steps_per_mm_y = steps_per_mm_y if steps_per_mm_y is not None else steps_per_mm
        self.max_steps = tuple(int(v) for v in max_steps)
        self.safety_margin_steps = int(safety_margin_steps)
        # Safe hard limits in steps: [margin, max - margin]
        self.limit_lo_steps = (0, 0)
        self.limit_hi_steps = (
            max(0, self.max_steps[0] - self.safety_margin_steps),
            max(0, self.max_steps[1] - self.safety_margin_steps),
        )
        # mm equivalents (for env, which works in mm)
        self.limit_lo = (0.0, 0.0)
        self.limit_hi = (
            self.limit_hi_steps[0] / self.steps_per_mm_x,
            self.limit_hi_steps[1] / self.steps_per_mm_y,
        )
        self.steps_per_mm = {
            'x': self.steps_per_mm_x,
            'y': self.steps_per_mm_y,
        }
        
        self.ser: Optional[serial.Serial] = None
        self.lock = threading.Lock()
        self.connected = False
        
        # Position tracking
        self.x_pos = 0.0
        self.y_pos = 0.0
        self._homed = False
    
    def find_port(self) -> Optional[str]:
        """Auto-detect Arduino port."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if "Arduino" in p.description or "ACM" in p.device or "USB" in p.device:
                return p.device
        return None
    
    def connect(self, port: Optional[str] = None, suppress_reconnect: bool = False) -> bool:
        """Connect to Arduino."""
        if self.connected:
            return True
        
        target_port = port or self.port or self.find_port()
        if not target_port:
            logger.error("No Arduino port found")
            return False
        
        try:
            self.ser = serial.Serial(target_port, self.baud, timeout=self.timeout)
            time.sleep(2)  # Wait for Arduino reset
            self.ser.reset_input_buffer()
            self.port = target_port
            self.connected = True
            logger.info(f"Connected to {target_port} @ {self.baud} baud")
            
            # Initialize
            self.send_command("s", 60, suppress_reconnect=suppress_reconnect)  # Default speed
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        if self.ser and self.ser.is_open:
            try:
                self.send_command("r", suppress_reconnect=True)  # Return to home
                time.sleep(1)
            except:
                pass
            self.ser.close()
        self.connected = False
        logger.info("Disconnected")
    
    def _reconnect(self):
        """Reconnect serial after a USB drop; re-home to resync position tracking."""
        if getattr(self, "_reconnecting", False):
            return None  # avoid recursion (home() -> send_command -> _reconnect)
        self._reconnecting = True
        try:
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None
            self.connected = False
            self._homed = False
            self.x_pos = 0.0
            self.y_pos = 0.0
            ok = self.connect(suppress_reconnect=True)
            if ok:
                try:
                    ok = self.home(suppress_reconnect=True)
                except OSError:
                    ok = False
            return ok
        finally:
            self._reconnecting = False

    def send_command(self, cmd: str, value: Optional[int] = None,
                     suppress_reconnect: bool = False) -> List[str]:
        """Send command and read response lines; auto-reconnect on USB drops."""
        if not self.connected or not self.ser:
            raise RuntimeError("Not connected")
        
        for attempt in range(3):
            if not self.connected or not self.ser:
                if suppress_reconnect:
                    raise RuntimeError("Not connected after drop")
                if not self._reconnect():
                    raise RuntimeError("Serial connection lost and reconnect failed")
            with self.lock:
                try:
                    if value is not None:
                        line = f"{cmd} {int(value)}\n"
                    else:
                        line = f"{cmd}\n"
                    
                    self.ser.write(line.encode())
                    self.ser.flush()
                    
                    responses = []
                    start = time.time()
                    while time.time() - start < self.timeout:
                        if self.ser.in_waiting:
                            resp = self.ser.readline().decode(errors="replace").strip()
                            if resp:
                                responses.append(resp)
                                # 'r' floods intermediate "Moving X/Y" lines; wait for the real completion
                                if cmd == "r":
                                    if "Rest position reached" in resp:
                                        break
                                elif "Moving" in resp or "Speed set" in resp or "Pen servo set" in resp:
                                    break
                        else:
                            time.sleep(0.01)
                    
                    return responses
                except OSError:
                    logger.warning(f"Serial dropped (attempt {attempt+1}) — reconnecting...")
            # outside lock: reconnect
            if suppress_reconnect or not self._reconnect():
                raise RuntimeError("Serial connection lost and reconnect failed")
        raise RuntimeError("Serial connection lost and reconnect failed")
    
    def move_x(self, steps: int) -> bool:
        """Move X motor by steps (relative)."""
        resp = self.send_command("x", steps)
        return len(resp) > 0 and "Moving X" in resp[0]
    
    def move_y(self, steps: int) -> bool:
        resp = self.send_command("y", steps)
        return len(resp) > 0 and "Moving Y" in resp[0]
    
    def move_both(self, steps: int) -> bool:
        resp = self.send_command("b", steps)
        return len(resp) > 0 and "Moving both" in resp[0]
    
    def move_to_mm(self, x_mm: float, y_mm: float) -> bool:
        """Move to absolute position in mm (clamped to safe step limits)."""
        if not self._homed:
            self.home()
        
        # Convert targets to steps, clamp hard to safe limits
        tx = int(round(x_mm * self.steps_per_mm_x))
        ty = int(round(y_mm * self.steps_per_mm_y))
        tx = min(max(tx, self.limit_lo_steps[0]), self.limit_hi_steps[0])
        ty = min(max(ty, self.limit_lo_steps[1]), self.limit_hi_steps[1])
        
        current_x_steps = int(round(self.x_pos * self.steps_per_mm_x))
        current_y_steps = int(round(self.y_pos * self.steps_per_mm_y))
        
        dx = tx - current_x_steps
        dy = ty - current_y_steps
        
        # Simple approach: move X then Y
        if dx != 0:
            self.send_command("x", dx)
            self.x_pos = tx / self.steps_per_mm_x
        if dy != 0:
            self.send_command("y", dy)
            self.y_pos = ty / self.steps_per_mm_y
        
        return True
    
    def home(self, suppress_reconnect: bool = False) -> bool:
        """Return to origin (0,0)."""
        resp = self.send_command("r", suppress_reconnect=suppress_reconnect)
        if resp and "Rest position reached" in resp[-1]:
            self.x_pos = 0.0
            self.y_pos = 0.0
            self._homed = True
            return True
        return False
    
    def set_speed(self, rpm: int) -> bool:
        resp = self.send_command("s", rpm)
        return len(resp) > 0 and "Speed set" in resp[0]
    
    def set_pen(self, angle: int) -> bool:
        """Set pen servo angle (0-180). Requires 'p' command in firmware."""
        resp = self.send_command("p", int(angle))
        return len(resp) > 0 and "Pen servo set" in resp[0]
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get current position in mm (estimated from step counts)."""
        return (self.x_pos, self.y_pos)
    
    def is_connected(self) -> bool:
        return self.connected and self.ser and self.ser.is_open
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class SimulatedHardware(HardwareInterface):
    """Drop-in replacement for testing without hardware."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connected = True
        self.ser = None  # No serial
    
    def connect(self, port: Optional[str] = None) -> bool:
        self.connected = True
        return True
    
    def disconnect(self):
        self.connected = False
    
    def send_command(self, cmd: str, value: Optional[int] = None) -> List[str]:
        time.sleep(0.01)  # Simulate latency
        if cmd == "p":
            return [f"Pen servo set to {value}"]
        return [f"Moving {cmd.upper()} {value} steps"]
    
    def move_to_mm(self, x_mm: float, y_mm: float) -> bool:
        self.x_pos = float(np.clip(x_mm, self.limit_lo[0], self.limit_hi[0]))
        self.y_pos = float(np.clip(y_mm, self.limit_lo[1], self.limit_hi[1]))
        return True
    
    def home(self) -> bool:
        self.x_pos = 0.0
        self.y_pos = 0.0
        self._homed = True
        return True


# For backwards compatibility
import numpy as np