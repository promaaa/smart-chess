import time
import busio
import digitalio
import board
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.ili9341 as ili9341

# --- CONFIGURATION PINS (Confirmé fonctionnel) ---
CS_PIN = digitalio.DigitalInOut(board.CE0)   # Pin 24
DC_PIN = digitalio.DigitalInOut(board.D25)   # Pin 22
RST_PIN = digitalio.DigitalInOut(board.D24)  # Pin 18 (Mapping regroupé)

# Vitesse standard (24MHz) maintenant que le câblage est validé
BAUDRATE = 24000000

# SPI Matériel
spi = busio.SPI(clock=board.SCK, MOSI=board.MOSI, MISO=board.MISO)

# --- HARD RESET ---
# Toujours utile pour garantir le démarrage de l'écran à froid
RST_PIN.direction = digitalio.Direction.OUTPUT
RST_PIN.value = True
time.sleep(0.1)
RST_PIN.value = False
time.sleep(0.1)
RST_PIN.value = True
time.sleep(0.1)

# --- INITIALISATION ---
disp = ili9341.ILI9341(
    spi,
    cs=CS_PIN,
    dc=DC_PIN,
    rst=RST_PIN,
    baudrate=BAUDRATE,
    width=320,
    height=240
)

# --- FONCTION DE CORRECTION BGR ---
# Votre écran est BGR, mais Pillow travaille en RGB.
# Cette fonction convertit (R, G, B) -> (B, G, R) pour l'affichage.
def color(r, g, b):
    return (b, g, r)

# Définition des couleurs corrigées
NOIR = color(0, 0, 0)
BLANC = color(255, 255, 255)
ROUGE = color(255, 0, 0)
VERT  = color(0, 255, 0)
BLEU  = color(0, 0, 255)

# --- DESSIN ---
if disp.rotation % 180 == 90:
    height = disp.width
    width = disp.height
else:
    width = disp.width
    height = disp.height

# Création de l'image de base
image = Image.new("RGB", (width, height))
draw = ImageDraw.Draw(image)

# Police
try:
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except OSError:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

print("Affichage du test final...")

while True:
    # --- TEST ROUGE ---
    # Fond Rouge (corrigé)
    draw.rectangle((0, 0, width, height), fill=ROUGE)
    # Carré central Blanc
    draw.rectangle((20, 20, width-20, height-20), outline=NOIR, fill=BLANC)
    
    text = "ROUGE OK"
    bbox = draw.textbbox((0, 0), text, font=font_large)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Texte en Rouge
    draw.text(((width-text_w)//2, (height-text_h)//2 - 20), text, font=font_large, fill=ROUGE)
    draw.text((30, height-50), "Vitesse: 24MHz", font=font_small, fill=NOIR)
    
    disp.image(image)
    time.sleep(2)

    # --- TEST BLEU ---
    # Fond Bleu (corrigé)
    draw.rectangle((0, 0, width, height), fill=BLEU)
    # Carré central Blanc
    draw.rectangle((20, 20, width-20, height-20), outline=NOIR, fill=BLANC)
    
    text = "BLEU OK"
    bbox = draw.textbbox((0, 0), text, font=font_large)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Texte en Bleu
    draw.text(((width-text_w)//2, (height-text_h)//2 - 20), text, font=font_large, fill=BLEU)
    draw.text((30, height-50), "Couleurs: BGR->RGB", font=font_small, fill=NOIR)
    
    disp.image(image)
    time.sleep(2)

    # --- TEST VERT ---
    draw.rectangle((0, 0, width, height), fill=VERT)
    draw.rectangle((20, 20, width-20, height-20), outline=NOIR, fill=BLANC)
    
    text = "VERT OK"
    bbox = draw.textbbox((0, 0), text, font=font_large)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    draw.text(((width-text_w)//2, (height-text_h)//2 - 20), text, font=font_large, fill=VERT)
    
    disp.image(image)
    time.sleep(2)