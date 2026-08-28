#include <AFMotor.h>
#include <Servo.h>

const int STEPS_PER_REV = 200; // typical for DVD steppers; adjust if yours differs

// Stepper 1 -> M1-M2 terminals (X), Stepper 2 -> M3-M4 terminals (Y)
AF_Stepper stepperX(STEPS_PER_REV, 1);
AF_Stepper stepperY(STEPS_PER_REV, 2);

// Pen servo on the L293D shield's onboard servo header
// SERVO 1 header -> D9, SERVO 2 header -> D10 (free pins, not used by AFMotor PWM)
int penServoPin = 10; // SERVO_1 header = D10 on this HW-130 clone
Servo penServo;

int currentSpeed = 60; // RPM - keep low when running from USB 5V

long xPos = 0; // current position of X motor (steps from home)
long yPos = 0; // current position of Y motor (steps from home)

void setup() {
  Serial.begin(9600);
  stepperX.setSpeed(currentSpeed);
  stepperY.setSpeed(currentSpeed);
  penServo.attach(penServoPin);
  penServo.write(90); // pen up position at startup
  printHelp();
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;
    char cmd = line.charAt(0);
    int value = line.substring(1).toInt();

    switch (cmd) {
      case 'x':
        moveOne(stepperX, xPos, 'X', value);
        break;
      case 'y':
        moveOne(stepperY, yPos, 'Y', value);
        break;
      case 'b':
        moveBoth(value);
        break;
      case 'r':
        rest();
        break;
      case 's':
        currentSpeed = constrain(value, 1, 200);
        stepperX.setSpeed(currentSpeed);
        stepperY.setSpeed(currentSpeed);
        Serial.print(F("Speed set to "));
        Serial.println(currentSpeed);
        break;
      case 'p':
        value = constrain(value, 0, 180);
        penServo.write(value);
        Serial.print(F("Pen servo set to "));
        Serial.println(value);
        break;
      case 'h':
      case '?':
        printHelp();
        break;
      default:
        Serial.println(F("Unknown command. Type h for help."));
    }
  }
}

void moveOne(AF_Stepper &s, long &pos, char name, int steps) {
  if (steps == 0) return;
  uint8_t dir = steps > 0 ? FORWARD : BACKWARD;
  s.step(abs(steps), dir, DOUBLE);
  pos += steps;
  Serial.print(F("Moving "));
  Serial.print(name);
  Serial.print(F(" "));
  Serial.print(abs(steps));
  Serial.println(F(" steps"));
}

void moveBoth(int steps) {
  if (steps == 0) return;
  uint8_t dir = steps > 0 ? FORWARD : BACKWARD;
  stepperX.step(abs(steps), dir, DOUBLE);
  stepperY.step(abs(steps), dir, DOUBLE);
  xPos += steps;
  yPos += steps;
  Serial.print(F("Moving both "));
  Serial.print(abs(steps));
  Serial.println(F(" steps"));
}

void rest() {
  Serial.println(F("Returning to rest (0)"));
  homeMotor(stepperX, xPos, 'X');
  homeMotor(stepperY, yPos, 'Y');
  Serial.println(F("Rest position reached"));
}

void homeMotor(AF_Stepper &s, long &pos, char name) {
  long steps = -pos;
  if (steps == 0) {
    Serial.print(name);
    Serial.println(F(" already at 0"));
    return;
  }
  uint8_t dir = steps > 0 ? FORWARD : BACKWARD;
  Serial.print(name);
  Serial.print(F(" back "));
  Serial.print(abs(steps));
  Serial.println(F(" steps"));
  s.step(abs(steps), dir, DOUBLE);
  pos = 0;
}

void printHelp() {
  Serial.println(F("Commands:"));
  Serial.println(F("  x <steps>   move X motor (+/- for direction)"));
  Serial.println(F("  y <steps>   move Y motor (+/- for direction)"));
  Serial.println(F("  b <steps>   move both motors together"));
  Serial.println(F("  r           rest - return both motors to 0 position"));
  Serial.println(F("  s <rpm>     set speed (e.g. 30-150)"));
  Serial.println(F("  p <angle>   set pen servo angle 0-180 (90=up, 45=down)"));
  Serial.println(F("  h           help"));
}