#!/usr/bin/env python3
"""Web UI to control the HW-130 L293D stepper shield over serial."""
import serial
import serial.tools.list_ports
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = "/dev/ttyACM1"
BAUD = 9600

ser = None
lock = threading.Lock()
current_port = None


def available_ports():
    return [p.device for p in serial.tools.list_ports.comports()
            if p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB", "COM"))]


def find_port():
    ports = available_ports()
    if current_port in ports:
        return current_port
    if PORT in ports:
        return PORT
    return ports[0] if ports else None


def connect(port=None):
    global ser, current_port
    if ser and ser.is_open:
        try:
            ser.close()
        except Exception:
            pass
        ser = None
    if port:
        current_port = port
    p = find_port()
    if p:
        try:
            ser = serial.Serial(p, BAUD, timeout=1)
            current_port = p
            return f"connected to {p}"
        except Exception as e:
            return f"connect failed: {e}"
    return "no port found"


def send(cmd, value):
    with lock:
        if not ser or not ser.is_open:
            return "NOT CONNECTED"
        try:
            ser.reset_input_buffer()
            ser.write(f"{cmd} {int(value)}\n".encode())
            lines = []
            while True:
                line = ser.readline().decode(errors="replace").strip()
                if not line:
                    break
                lines.append(line)
            return " | ".join(lines) or "ok"
        except Exception as e:
            return f"error: {e}"


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stepper Control</title>
<style>
  body { font-family: sans-serif; margin: 20px; }
  h1 { font-size: 20px; }
  .panel { border: 1px solid #ccc; border-radius: 8px; padding: 15px;
           margin-bottom: 15px; display: inline-block; vertical-align: top; width: 300px; }
  .slider { width: 260px; }
  .val { font-weight: bold; }
  button { margin: 3px; padding: 8px 14px; font-size: 15px; cursor: pointer; }
  #log { background:#111; color:#0f0; font-family: monospace; padding: 8px;
         height: 120px; overflow-y: auto; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>Stepper Control — HW130 (2x DVD stepper)</h1>

<div style="margin-bottom:15px">
  Port:
  <select id="portsel"></select>
  <button onclick="refreshPorts()">Refresh</button>
  <button onclick="connectPort()">Connect</button>
  <span id="status" style="font-weight:bold"></span>
</div>

<div class="panel" style="background:#fff3e0">
  <h2>X motor (M1–M2)</h2>
  <div>steps: <span class="val" id="xval">200</span></div>
  <input type="range" class="slider" id="x" min="1" max="1000" value="200"
         oninput="document.getElementById('xval').textContent=this.value">
  <br>
  <button onclick="go('x', +document.getElementById('x').value)">X Up</button>
  <button onclick="go('x', -document.getElementById('x').value)">X Down</button>
</div>

<div class="panel" style="background:#e3f2fd">
  <h2>Y motor (M3–M4)</h2>
  <div>steps: <span class="val" id="yval">200</span></div>
  <input type="range" class="slider" id="y" min="1" max="1000" value="200"
         oninput="document.getElementById('yval').textContent=this.value">
  <br>
  <button onclick="go('y', +document.getElementById('y').value)">Y Up</button>
  <button onclick="go('y', -document.getElementById('y').value)">Y Down</button>
</div>

<div class="panel" style="background:#e8f5e9">
  <h2>Both motors</h2>
  <div>steps: <span class="val" id="bval">200</span></div>
  <input type="range" class="slider" id="b" min="1" max="1000" value="200"
         oninput="document.getElementById('bval').textContent=this.value">
  <br>
  <button onclick="go('b', +document.getElementById('b').value)">Both Up</button>
  <button onclick="go('b', -document.getElementById('b').value)">Both Down</button>
  <br>
  <button onclick="go('r', 0)" style="background:#ffcdd2">REST (return to 0)</button>
</div>

<div class="panel" style="background:#f3e5f5">
  <h2>Speed</h2>
  <div>RPM: <span class="val" id="sval">60</span></div>
  <input type="range" class="slider" id="s" min="5" max="200" value="60"
         oninput="document.getElementById('sval').textContent=this.value">
  <br>
  <button onclick="go('s', +document.getElementById('s').value)">Apply Speed</button>
</div>

<h3>Output</h3>
<div id="log"></div>

<script>
function go(cmd, val) {
  fetch('/cmd?c=' + cmd + '&v=' + val).then(r => r.text()).then(t => {
    document.getElementById('log').textContent += t + '\\n';
  });
}
function refreshPorts() {
  fetch('/ports').then(r => r.json()).then(list => {
    const sel = document.getElementById('portsel');
    sel.innerHTML = '';
    list.forEach(p => {
      const o = document.createElement('option');
      o.value = p; o.textContent = p;
      sel.appendChild(o);
    });
  });
}
function connectPort() {
  const p = document.getElementById('portsel').value;
  fetch('/connect?p=' + encodeURIComponent(p)).then(r => r.text()).then(t => {
    document.getElementById('status').textContent = t;
    refreshStatus();
  });
}
function refreshStatus() {
  fetch('/status').then(r => r.text()).then(t => {
    document.getElementById('status').textContent = t;
  });
}
refreshPorts();
refreshStatus();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/cmd"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            cmd = q.get("c", ["x"])[0]
            val = int(q.get("v", [0])[0])
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(send(cmd, val).encode())
        elif self.path.startswith("/connect"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            port = q.get("p", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(connect(port).encode())
        elif self.path == "/ports":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(available_ports()).encode())
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            state = "connected" if ser and ser.is_open else "disconnected"
            self.wfile.write(f"{state} @ {current_port}".encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(connect())
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Open http://localhost:8000 in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if ser:
            ser.close()