#!/bin/bash
# Option B - Workflow Complet
# ============================

echo "🚀 OPTION B: TUNING RAPIDE AVEC DATASET"
echo "========================================"
echo ""

# Déterminer le bon interpréteur Python (celui du venv)
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -f "/Users/promaa/Documents/code/smart-chess/venv/bin/python" ]; then
    PYTHON="/Users/promaa/Documents/code/smart-chess/venv/bin/python"
else
    PYTHON="python3"
fi

echo "Using Python: $PYTHON"
echo ""

# Étape 1: Génération du dataset (30-60 min)
echo "📊 Étape 1/3: Génération du dataset..."
echo "Temps estimé: 30-60 minutes"
echo ""
read -p "Appuyez sur ENTRÉE pour démarrer..."

cd /Users/promaa/Documents/code/smart-chess/ia_marc/V2
$PYTHON tuning/generate_dataset_quick.py

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de la génération du dataset"
    exit 1
fi

echo ""
echo "✅ Dataset généré avec succès !"
echo ""

# Étape 2: Tuning (5-10 min)
echo "⚙️  Étape 2/3: Optimisation des poids..."
echo "Temps estimé: 5-10 minutes"
echo ""
read -p "Appuyez sur ENTRÉE pour lancer le tuning..."

cd tuning
$PYTHON run_tuner.py

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors du tuning"
    exit 1
fi

echo ""
echo "✅ Tuning terminé !"
echo ""

# Étape 3: Instructions pour appliquer
echo "📝 Étape 3/3: Application des poids"
echo "===================================="
echo ""
echo "Les nouveaux poids sont dans: tuning/optimized_weights.json"
echo ""
echo "Pour les appliquer:"
echo "  1. Ouvrir ia_marc/V2/engine_brain.py"
echo "  2. Modifier les lignes 27-28 avec les nouvelles valeurs"
echo "  3. Tester avec: python3 ../ai_comparison/compare_v2_stockfish.py"
echo ""
echo "✅ TUNING OPTION B TERMINÉ !"
