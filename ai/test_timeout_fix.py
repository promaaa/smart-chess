#!/usr/bin/env python3
"""
Test de vérification du timeout fixé dans le null-move engine
"""

import sys
import os
import time

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Chess import Chess
from Null_move_AI.null_move_engine import NullMovePruningEngine

def test_timeout_fix():
    """
    Test que le timeout fonctionne maintenant correctement
    """
    print("🔧 === TEST DU FIX TIMEOUT === 🔧")
    print("⏱️  Temps limite: 10 secondes (doit s'arrêter à temps)")
    print()
    
    chess = Chess()
    engine = NullMovePruningEngine(
        max_time=10.0,  # 10 secondes seulement
        max_depth=20,
        null_move_enabled=True,
        null_move_R=2,
        null_move_min_depth=3,
        tt_size=1000000
    )
    
    print("🚀 Démarrage recherche (devrait s'arrêter en ~10s)...")
    start_time = time.time()
    
    try:
        best_move = engine.get_best_move_with_time_limit(chess)
        actual_time = time.time() - start_time
        
        print(f"✅ Recherche terminée!")
        print(f"⏱️  Temps réel: {actual_time:.2f}s")
        print(f"🎯 Coup: {engine._format_move(best_move) if best_move else 'None'}")
        print(f"📊 Nœuds: {engine.nodes_evaluated:,}")
        
        # Vérification du timeout
        if actual_time <= 12.0:  # 2s de marge
            print(f"✅ TIMEOUT CORRIGÉ! ({actual_time:.1f}s ≤ 12s)")
            return True
        else:
            print(f"❌ TIMEOUT ENCORE CASSÉ! ({actual_time:.1f}s > 12s)")
            return False
            
    except Exception as e:
        actual_time = time.time() - start_time
        print(f"❌ Erreur après {actual_time:.2f}s: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Test du fix de timeout pour null-move engine")
    print("=" * 50)
    
    success = test_timeout_fix()
    
    if success:
        print(f"\n🎉 SUCCÈS: Le timeout fonctionne maintenant!")
        print(f"✅ Les tests de 60s ne dépasseront plus la limite")
    else:
        print(f"\n😓 ÉCHEC: Le timeout ne fonctionne toujours pas")
        print(f"❌ Il faut investiguer davantage")
    
    print(f"\n🏁 Test terminé")