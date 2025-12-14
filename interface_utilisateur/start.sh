#!/bin/bash
# Smart Chess - Script de lancement
# Lance le serveur avec l'IA Marc V2 et ouvre l'interface

cd "$(dirname "$0")"

echo "╔══════════════════════════════════════╗"
echo "║       Smart Chess - Démarrage        ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé. Installez Python 3."
    exit 1
fi

# Activer le virtual environment
VENV_PATH="$(dirname "$0")/../venv"
if [ -d "$VENV_PATH" ]; then
    echo "🐍 Activation du virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "⚠️  Virtual environment non trouvé, création..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Vérifier les dépendances
python3 -c "import chess" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installation des dépendances..."
    pip install python-chess
fi

# Lancer le serveur
echo "🚀 Lancement du serveur..."
echo ""

# Ouvrir le navigateur après 3 secondes
(sleep 3 && open "http://localhost:8080" 2>/dev/null || xdg-open "http://localhost:8080" 2>/dev/null) &

# Démarrer le serveur
python3 server.py
