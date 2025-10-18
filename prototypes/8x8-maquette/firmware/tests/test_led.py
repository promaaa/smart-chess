import time
import board
import busio
from adafruit_tca9548a import TCA9548A
from adafruit_is31fl3731.matrix import Matrix

# Initialisation du bus I²C principal du Raspberry Pi
i2c = busio.I2C(board.SCL, board.SDA)

# Initialisation du multiplexeur TCA9548A
tca = TCA9548A(i2c)

# Sélection du canal 4 (correspondant à SD4/SC4)
i2c_led = tca[4]

# Initialisation du module IS31FL3731 sur ce canal
display = Matrix(i2c_led)

# --- Séquence de test ---
display.fill(0)

# LED centrale
display.pixel(4, 4, 255)
time.sleep(1)

# Diagonale
display.fill(0)
for i in range(9):
    display.pixel(i, i, 180)
    time.sleep(0.1)

# Balayage horizontal
display.fill(0)
for x in range(9):
    display.fill(0)
    for y in range(9):
        display.pixel(x, y, 255)
    time.sleep(0.1)

# Variation d’intensité
for brightness in range(0, 256, 32):
    display.fill(brightness)
    time.sleep(0.2)

display.fill(0)
print("Test terminé sur canal 4.")
