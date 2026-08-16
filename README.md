# 3D Writing Robot

Control two DVD-player stepper motors (X and Y axis) using an Arduino Uno with an **HW-130 L293D motor driver shield**.

## Hardware

- Arduino Uno
- HW-130 L293D motor driver shield (2x L293D + 74HC595)
- 2x DVD player stepper motors (4-wire bipolar, ~200 steps/rev)

### Wiring

- Stepper 1 (X axis): coil A -> M1, coil B -> M2 terminals
- Stepper 2 (Y axis): coil A -> M3, coil B -> M4 terminals
- Power: USB 5V (small DVD steppers) or external 7-12V (recommended)

## Files

- `dvd_stepper_serial.ino` — Arduino firmware
- `stepper_web.py` — Web control UI (sliders + buttons over serial)

## Arduino firmware

Upload `dvd_stepper_serial.ino` to the Uno (requires the **Adafruit Motor Shield library** / AFMotor).

```
arduino-cli lib install "Adafruit Motor Shield library"
arduino-cli compile --fqbn arduino:avr:uno dvd_stepper_serial
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno dvd_stepper_serial
```

### Serial commands (9600 baud)

| Command | Action |
|---|---|
| `x <steps>` | Move X motor (+/- direction) |
| `y <steps>` | Move Y motor (+/- direction) |
| `b <steps>` | Move both motors together |
| `r` | Rest — return both motors to 0 position |
| `s <rpm>` | Set speed (e.g. 30-150) |
| `h` | Help |

## Web UI

```
python3 stepper_web.py
```

Open http://localhost:8000

Features:
- Sliders for X, Y, and Both step amounts (1-1000)
- Up/Down buttons per motor and for both
- REST button to return to the 0 (home) position
- Speed slider with Apply Speed
- Live output log
- USB port selector (Refresh + Connect) for when the port changes

Requires `pyserial` (`pip install pyserial`).