# test_reeds_mux_ident.py
from smbus2 import SMBus
import time

# --- Constantes ---
TCA_ADDR = 0x70  # Adresse par défaut du TCA9548A
MCP_ADDRS = [0x20 + i for i in range(8)]  # 0x20..0x27 possibles
BUSNUM = 1

# Registres MCP23017
GPIOA, GPIOB = 0x12, 0x13
IODIRA, IODIRB = 0x00, 0x01
GPPUA, GPPUB = 0x0C, 0x0D


# --- Fonctions ---
def tca_select(bus, ch):
    """Sélectionne un canal sur le multiplexeur TCA9548A."""
    if ch is None:
        bus.write_byte(TCA_ADDR, 0x00)  # Désactive tous les canaux
    else:
        bus.write_byte(TCA_ADDR, 1 << ch)


def init_mcp_if_present(bus, addr):
    """Initialise un MCP23017 s'il est présent à l'adresse donnée."""
    try:
        # Configuration : tous les pins en entrée avec résistances de pull-up internes
        bus.write_byte_data(addr, IODIRA, 0xFF)
        bus.write_byte_data(addr, IODIRB, 0xFF)
        bus.write_byte_data(addr, GPPUA, 0xFF)
        bus.write_byte_data(addr, GPPUB, 0xFF)
        return True
    except OSError:
        return False


def read_mcp_gpio(bus, addr):
    """Lit les états des ports GPIOA et GPIOB d'un MCP23017."""
    try:
        a = bus.read_byte_data(addr, GPIOA)
        b = bus.read_byte_data(addr, GPIOB)
        return a, b
    except OSError:
        return None


def pretty_bits(val):
    """Formate un entier 8 bits en une chaîne binaire (ex: 11111110)."""
    return format(val, "08b")


def get_active_reeds(value, prefix):
    """
    Identifie les reeds actifs (bits à 0) dans une valeur de 8 bits.
    Retourne une liste de noms de reeds, par ex. ['A3', 'A7'].
    `prefix` est 'A' pour GPIOA et 'B' pour GPIOB.
    """
    active_list = []
    for i in range(8):
        # Vérifie si le i-ème bit est à 0 (contact fermé)
        if not (value & (1 << i)):
            active_list.append(f"{prefix}{i}")
    return active_list


# --- Programme principal ---
with SMBus(BUSNUM) as bus:
    # Vérifie si le TCA9548A est bien présent
    try:
        bus.read_byte(TCA_ADDR)
    except OSError:
        print(
            "Erreur : TCA9548A introuvable à l'adresse 0x70. Vérifiez le câblage et l'alimentation."
        )
        raise SystemExit

    print("TCA9548A détecté. Scan des canaux 0 à 7 pour les MCP23017 (0x20-0x27)...")
    print("-" * 70)

    try:
        while True:
            found_any_mcp_overall = False
            for ch in range(8):
                tca_select(bus, ch)
                time.sleep(0.01)

                for addr in MCP_ADDRS:
                    # On tente de lire le MCP. L'initialisation se fait une seule fois au début.
                    # Note : dans un script en boucle, l'init devrait être faite une seule fois.
                    # Pour un simple test, on peut la laisser ici.

                    if init_mcp_if_present(bus, addr):
                        found_any_mcp_overall = True
                        gpio_states = read_mcp_gpio(bus, addr)

                        if gpio_states:
                            a, b = gpio_states

                            # On ne traite que si un état a changé par rapport à "tout ouvert" (0xFF)
                            if a != 0xFF or b != 0xFF:
                                active_a = get_active_reeds(a, "A")
                                active_b = get_active_reeds(b, "B")
                                all_active = active_a + active_b

                                # Prépare la chaîne de caractères pour l'affichage
                                status_msg = (
                                    f"Activé(s) : {', '.join(all_active)}"
                                    if all_active
                                    else "Aucun reed activé"
                                )
                                print(
                                    f"Canal {ch} | MCP 0x{addr:02X} | A={pretty_bits(a)} B={pretty_bits(b)} | {status_msg}"
                                )

            if not found_any_mcp_overall:
                print(
                    "Aucun MCP23017 n'a été détecté sur aucun canal. Vérifiez le câblage."
                )
                break  # Sortir si aucun MCP n'est jamais trouvé

            print("\nScan en continu... Approchez un aimant. (Ctrl+C pour arrêter)")
            time.sleep(0.5)  # Pause entre les scans complets

    except KeyboardInterrupt:
        print("\nProgramme arrêté par l'utilisateur.")
    finally:
        tca_select(bus, None)  # Désélectionne tous les canaux en sortant
