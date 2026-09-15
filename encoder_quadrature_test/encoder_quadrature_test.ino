/*
  Quadrature encoder speed and direction test for ESP32.

  C1 wiring already validated:
    Left  C1 -> GPIO 4
    Right C1 -> GPIO 5

  Suggested C2 wiring:
    Left  C2 -> GPIO 16
    Right C2 -> GPIO 17

  Connect encoder VCC to 3.3 V and GND to GND.
  Open Serial Monitor at 115200 baud.

  The previous C1-only test measured about 980 falling edges per wheel
  revolution. With both channels and x4 decoding, use approximately
  3920 counts per revolution. Confirm this with a measured full turn.
*/

#define LEFT_C1_PIN  4
#define RIGHT_C1_PIN 5
#define LEFT_C2_PIN  16
#define RIGHT_C2_PIN 17

// Approximate x4 counts/revolution: 980 C1 falling edges x 4.
const float COUNTS_PER_REV = 3920.0f;
const unsigned long REPORT_INTERVAL_MS = 100;

volatile long leftCount = 0;
volatile long rightCount = 0;
volatile uint8_t leftState = 0;
volatile uint8_t rightState = 0;

// Quadrature transition table: previous state -> current state.
const int8_t QUADRATURE_DELTA[16] = {
   0, -1,  1,  0,
   1,  0,  0, -1,
  -1,  0,  0,  1,
   0,  1, -1,  0
};

void IRAM_ATTR onLeftChange() {
  uint8_t current = (digitalRead(LEFT_C1_PIN) << 1) | digitalRead(LEFT_C2_PIN);
  leftCount += QUADRATURE_DELTA[(leftState << 2) | current];
  leftState = current;
}

void IRAM_ATTR onRightChange() {
  uint8_t current = (digitalRead(RIGHT_C1_PIN) << 1) | digitalRead(RIGHT_C2_PIN);
  rightCount += QUADRATURE_DELTA[(rightState << 2) | current];
  rightState = current;
}

void setup() {
  Serial.begin(115200);

  // INPUT_PULLUP is appropriate for open-collector encoder outputs.
  // If your module has an active push-pull output, INPUT also works.
  pinMode(LEFT_C1_PIN, INPUT_PULLUP);
  pinMode(LEFT_C2_PIN, INPUT_PULLUP);
  pinMode(RIGHT_C1_PIN, INPUT_PULLUP);
  pinMode(RIGHT_C2_PIN, INPUT_PULLUP);

  leftState = (digitalRead(LEFT_C1_PIN) << 1) | digitalRead(LEFT_C2_PIN);
  rightState = (digitalRead(RIGHT_C1_PIN) << 1) | digitalRead(RIGHT_C2_PIN);

  attachInterrupt(digitalPinToInterrupt(LEFT_C1_PIN), onLeftChange, CHANGE);
  attachInterrupt(digitalPinToInterrupt(LEFT_C2_PIN), onLeftChange, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_C1_PIN), onRightChange, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_C2_PIN), onRightChange, CHANGE);

  Serial.println("Quadrature encoder test started.");
  Serial.println("Turn each wheel by hand, then test motor movement.");
  Serial.println("Format: total counts | counts/s | RPM | direction");
}

void loop() {
  static unsigned long previousReport = millis();
  static long previousLeft = 0;
  static long previousRight = 0;

  unsigned long now = millis();
  if (now - previousReport < REPORT_INTERVAL_MS) {
    delay(1);
    return;
  }

  noInterrupts();
  long currentLeft = leftCount;
  long currentRight = rightCount;
  interrupts();

  float intervalSeconds = (now - previousReport) / 1000.0f;
  float leftCountsPerSecond = (currentLeft - previousLeft) / intervalSeconds;
  float rightCountsPerSecond = (currentRight - previousRight) / intervalSeconds;
  float leftRpm = leftCountsPerSecond * 60.0f / COUNTS_PER_REV;
  float rightRpm = rightCountsPerSecond * 60.0f / COUNTS_PER_REV;

  Serial.print("L: ");
  Serial.print(currentLeft);
  Serial.print(" | ");
  Serial.print(leftCountsPerSecond, 1);
  Serial.print(" cps | ");
  Serial.print(leftRpm, 2);
  Serial.print(" RPM | ");
  Serial.print(leftCountsPerSecond >= 0.0f ? "FORWARD" : "REVERSE");
  Serial.print("    R: ");
  Serial.print(currentRight);
  Serial.print(" | ");
  Serial.print(rightCountsPerSecond, 1);
  Serial.print(" cps | ");
  Serial.print(rightRpm, 2);
  Serial.print(" RPM | ");
  Serial.println(rightCountsPerSecond >= 0.0f ? "FORWARD" : "REVERSE");

  previousLeft = currentLeft;
  previousRight = currentRight;
  previousReport = now;
}
