import time
import board
import busio
from adafruit_tca9548a import TCA9548A
from adafruit_is31fl3731.matrix import Matrix
from adafruit_mcp230xx.mcp23017 import MCP23017

# --- Mappings configurables ---
# Reed sensor mapping: dict de (mcp_channel, pin) -> (row, col)
# mcp_channel: 0 à 3 (canaux du multiplexeur pour les MCP)
# pin: 0 à 15 (0-7 pour GPIOA, 8-15 pour GPIOB)
# row, col: 0 à 7 pour l'échiquier 8x8
# Modifiez ce dict pour remapper les reed sensors à leurs positions réelles.
reed_to_pos = {}
for ch in range(4):
    row_base = ch * 2
    for col in range(8):
        reed_to_pos[(ch, col)] = (row_base, col)  # GPIOA: pins 0-7 -> row row_base
        reed_to_pos[(ch, col + 8)] = (
            row_base + 1,
            col,
        )  # GPIOB: pins 8-15 -> row row_base+1

# Position to LED mapping: dict de (row, col) -> (led_x, led_y)
# row, col: 0 à 7 pour l'échiquier
# led_x, led_y: 0 à 8 pour la matrice LED 9x9 (ajustez si nécessaire, par ex. pour ignorer une ligne/colonne)
# Modifiez ce dict pour remapper les positions de l'échiquier aux coordonnées LED réelles.
pos_to_led = {}
for row in range(8):
    for col in range(8):
        pos_to_led[(row, col)] = (col, row)  # Par défaut: led_x = col, led_y = row

# Initialisation du bus I²C principal
i2c = busio.I2C(board.SCL, board.SDA)

# Initialisation du multiplexeur TCA9548A
tca = TCA9548A(i2c)

# Initialisation du module IS31FL3731 sur le canal 4
i2c_led = tca[4]
display = Matrix(i2c_led)

# Initialisation des MCP23017 (assumés sur canaux 0-3 à l'adresse 0x20)
mcps = []
for ch in range(4):
    try:
        mcp = MCP23017(tca[ch], address=0x20)
        # Configuration : tous les pins en entrée avec pull-ups
        mcp.iodir = 0xFFFF
        mcp.gppu = 0xFFFF
        mcps.append(mcp)
        print(f"MCP23017 initialisé sur canal {ch}.")
    except (OSError, ValueError):
        print(f"Aucun MCP23017 détecté sur canal {ch}.")

if not mcps:
    raise SystemExit("Aucun MCP23017 détecté. Vérifiez le câblage.")

# --- Séquence de test LED ---
display.fill(0)

# LED centrale (ajustez si nécessaire pour votre mapping LED)
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
print("Séquence de test LED terminée.")

# --- Boucle principale pour détection et affichage ---
previous = set()
potential_start = None
last_start = None
last_end = None

while True:
    current = set()
    for ch, mcp in enumerate(mcps):
        gpio = mcp.gpio
        for pin in range(16):
            if not (gpio & (1 << pin)):  # Bit à 0 : reed actif (aimant détecté)
                pos = reed_to_pos.get((ch, pin))
                if pos and 0 <= pos[0] <= 7 and 0 <= pos[1] <= 7:
                    current.add(pos)

    # Détection des changements pour identifier les coups
    removed = previous - current
    added = current - previous

    if removed and not added:
        if len(removed) == 1:
            potential_start = list(removed)[0]
            print(f"Pièce enlevée de {potential_start}")
        else:
            print(f"Multiples enlevés : {removed}")
    elif added and not removed:
        if len(added) == 1 and potential_start:
            end = list(added)[0]
            if end != potential_start:
                last_start = potential_start
                last_end = end
                print(f"Coup détecté : de {last_start} à {last_end}")
            potential_start = None
        else:
            print(f"Ajout sans départ ou multiple : {added}")
    elif removed or added:
        print(f"Changement complexe : enlevés {removed}, ajoutés {added}")

    # Affichage : éteint tout, allume les positions avec aimants + cases de départ/arrivée du dernier coup
    display.fill(0)
    for pos in current:
        led_pos = pos_to_led.get(pos)
        if led_pos:
            display.pixel(led_pos[0], led_pos[1], 255)
    if last_start:
        led_pos = pos_to_led.get(last_start)
        if led_pos:
            display.pixel(led_pos[0], led_pos[1], 255)
    if last_end:
        led_pos = pos_to_led.get(last_end)
        if led_pos:
            display.pixel(led_pos[0], led_pos[1], 255)

    previous = current.copy()
    time.sleep(0.1)  # Pause pour éviter une surcharge CPU, ajustable
