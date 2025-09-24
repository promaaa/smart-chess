#include <Arduino.h>

const uint8_t ledPins[9] = {
  2,3,4,
  5,6,7,
  8,9,10
};
// Index LED: 0=L00,1=L01,2=L02, 3=L10,4=L11,5=L12, 6=L20,7=L21,8=L22

const uint8_t reedPins[4] = {A0, A1, A2, A3};

// 0=S00 (square en haut à gauche), 1=S01 (square en haut à droite), 2=S10 (square en bas à gauche), 3=S11 (square en bas à droite)

// Coins LED par case (indices dans ledPins)
const uint8_t squareCorners[4][4] = {
  {0,1,3,4}, // S00 -> L00 L01 L10 L11
  {1,2,4,5}, // S01 -> L01 L02 L11 L12
  {3,4,6,7}, // S10 -> L10 L11 L20 L21
  {4,5,7,8}  // S11 -> L11 L12 L21 L22
};

bool readReedStable(uint8_t pin) {
  // INPUT_PULLUP: fermé => LOW (actif)
  if (digitalRead(pin) == LOW) {
    delay(20); // anti-rebond simple
    return digitalRead(pin) == LOW;
  }
  return false;
}

void setAllLeds(bool on) {
  for (uint8_t i=0;i<9;i++) digitalWrite(ledPins[i], on ? HIGH : LOW);
}

void startupTest() {
  // Balayage rapide pour vérifier le câblage
  setAllLeds(false);
  for (uint8_t i=0;i<9;i++) { digitalWrite(ledPins[i], HIGH); delay(80); }
  for (int i=8;i>=0;i--)    { digitalWrite(ledPins[i], LOW);  delay(40); }
}

void setup() {
  for (uint8_t i=0;i<9;i++) { pinMode(ledPins[i], OUTPUT); digitalWrite(ledPins[i], LOW); }
  for (uint8_t i=0;i<4;i++) pinMode(reedPins[i], INPUT_PULLUP);
  startupTest();
}

void loop() {
  // Calcul de l’union des coins à allumer
  bool ledState[9] = {false,false,false,false,false,false,false,false,false};

  for (uint8_t s=0; s<4; s++) {
    if (readReedStable(reedPins[s])) {
      for (uint8_t k=0; k<4; k++) {
        ledState[squareCorners[s][k]] = true;
      }
    }
  }

  // Mise à jour des LED
  for (uint8_t i=0;i<9;i++) digitalWrite(ledPins[i], ledState[i] ? HIGH : LOW);

  delay(5);
}
