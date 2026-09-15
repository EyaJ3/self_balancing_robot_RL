/*
  Standalone encoder count test for ESP32.
  No Raspberry Pi connection needed for this step -- just spin each wheel by
  hand and watch the Serial Monitor to check the count is consistent across
  identical full turns.

  Wiring:
    Left  encoder DO -> GPIO 4   (VCC -> 3.3V, GND -> GND)
    Right encoder DO -> GPIO 5   (VCC -> 3.3V, GND -> GND)

  Upload via Arduino IDE (board: "ESP32 Dev Module"), then open
  Serial Monitor at 115200 baud.
*/

#define LEFT_ENCODER_PIN  4
#define RIGHT_ENCODER_PIN 5

volatile long leftCount = 0;
volatile long rightCount = 0;

// Minimum microseconds between counted pulses, to reject electrical bounce/noise.
// Lower this if you spin the wheel fast and see counts being missed;
// raise it if you see extra counts from a single slow pulse edge.
const unsigned long DEBOUNCE_US = 500;

volatile unsigned long lastLeftMicros = 0;
volatile unsigned long lastRightMicros = 0;

void IRAM_ATTR onLeftPulse() {
  unsigned long now = micros();
  if (now - lastLeftMicros > DEBOUNCE_US) {
    leftCount++;
    lastLeftMicros = now;
  }
}

void IRAM_ATTR onRightPulse() {
  unsigned long now = micros();
  if (now - lastRightMicros > DEBOUNCE_US) {
    rightCount++;
    lastRightMicros = now;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LEFT_ENCODER_PIN, INPUT);
  pinMode(RIGHT_ENCODER_PIN, INPUT);

  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_PIN), onLeftPulse, FALLING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENCODER_PIN), onRightPulse, FALLING);

  Serial.println("Encoder test started.");
  Serial.println("Spin each wheel by hand, one full turn at a time, and compare counts.");
}

void loop() {
  static long lastLeft = -1;
  static long lastRight = -1;

  // Only print when something changed, so the Serial Monitor stays readable.
  if (leftCount != lastLeft || rightCount != lastRight) {
    noInterrupts();
    long l = leftCount;
    long r = rightCount;
    interrupts();

    Serial.print("Left: ");
    Serial.print(l);
    Serial.print("   Right: ");
    Serial.println(r);

    lastLeft = l;
    lastRight = r;
  }

  delay(20);
}
