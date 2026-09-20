/*
  ESP32 encoder telemetry.

  The ESP32 only reads the encoders. Motor control remains on the Raspberry Pi.
  Connect the ESP32 to the Raspberry Pi with 3.3 V UART; telemetry is sent as CSV:
    ENC,millis,left_count,right_count,left_rad_s,right_rad_s

  Encoder wiring:
    Left  C1 -> GPIO 4,  C2 -> GPIO 16
    Right C1 -> GPIO 5,  C2 -> GPIO 17

  UART wiring:
    ESP32 GPIO 33 (TX2) -> Raspberry Pi GPIO 15 (RXD)
    ESP32 GPIO 32 (RX2) <- Raspberry Pi GPIO 14 (TXD)
    ESP32 GND          -> Raspberry Pi GND
*/

#define LEFT_C1_PIN  4
#define RIGHT_C1_PIN 5
#define LEFT_C2_PIN  16
#define RIGHT_C2_PIN 17
#define ESP_UART_RX  32
#define ESP_UART_TX  33

const unsigned long REPORT_INTERVAL_MS = 50;
const float COUNTS_PER_REV = 3948.0f;
const float LEFT_ENCODER_SIGN = -1.0f;
const float RIGHT_ENCODER_SIGN = 1.0f;
const int8_t QUADRATURE_DELTA[16] = {
   0, -1,  1,  0,
   1,  0,  0, -1,
  -1,  0,  0,  1,
   0,  1, -1,  0
};

volatile long leftCount = 0;
volatile long rightCount = 0;
volatile uint8_t leftState = 0;
volatile uint8_t rightState = 0;

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
  Serial2.begin(115200, SERIAL_8N1, ESP_UART_RX, ESP_UART_TX);
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
}

void loop() {
  static unsigned long previousReport = 0;
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

  float dt = (now - previousReport) / 1000.0f;
  float leftCps = LEFT_ENCODER_SIGN * (currentLeft - previousLeft) / dt;
  float rightCps = RIGHT_ENCODER_SIGN * (currentRight - previousRight) / dt;
  float leftRadPerSecond = leftCps * 2.0f * PI / COUNTS_PER_REV;
  float rightRadPerSecond = rightCps * 2.0f * PI / COUNTS_PER_REV;

  Serial2.print("ENC,");
  Serial2.print(now);
  Serial2.print(",");
  Serial2.print(currentLeft);
  Serial2.print(",");
  Serial2.print(currentRight);
  Serial2.print(",");
  Serial2.print(leftRadPerSecond, 4);
  Serial2.print(",");
  Serial2.println(rightRadPerSecond, 4);

  previousLeft = currentLeft;
  previousRight = currentRight;
  previousReport = now;
}
