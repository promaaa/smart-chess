#!/bin/bash
# Smart Chess - Interface PvP Remote
# Script de démarrage

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../venv"

echo "╔══════════════════════════════════════╗"
echo "║    Smart Chess - PvP Remote          ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Vérifier le virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment non trouvé, création..."
    python3 -m venv "$VENV_DIR"
fi

# Activer le venv
source "$VENV_DIR/bin/activate"

# Installer les dépendances
echo "📦 Vérification des dépendances..."
pip install -q python-chess websockets

# Déterminer le mode
MODE=""
if [[ "$1" == "--simulation" ]] || [[ "$1" == "-s" ]]; then
    MODE="--simulation"
    echo ""
    echo "🎮 Mode SIMULATION activé (sans plateau physique)"
else
    echo ""
    echo "🔌 Mode HARDWARE (avec plateau physique)"
    echo "   Utilisez --simulation ou -s pour tester sans matériel"
fi

echo ""
echo "🚀 Lancement du serveur..."
echo ""

cd "$SCRIPT_DIR"
python3 server.py $MODE
