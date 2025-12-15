#!/usr/bin/env python3
"""
Smart Chess - Serveur Backend pour l'interface web
Connecte l'interface utilisateur à l'IA Marc V2
"""

import sys
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Ajouter le chemin vers IA Marc V2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai', 'ia_marc', 'V2'))

try:
    import chess
    from engine_main import ChessEngine
    from engine_config import DIFFICULTY_LEVELS, PERSONALITIES
    print("IA Marc V2 chargée avec succès")
except ImportError as e:
    print(f"Erreur d'import IA Marc: {e}")
    print("Assurez-vous que les dépendances sont installées: pip install python-chess")
    sys.exit(1)

# Instance globale du moteur
engine = None
current_level = "LEVEL5"


def init_engine(level="LEVEL5"):
    """Initialise le moteur d'échecs"""
    global engine, current_level
    
    try:
        engine = ChessEngine(verbose=True)
        engine.set_level(level)
        current_level = level
        
        # Essayer de charger le livre d'ouvertures Polyglot
        book_path = os.path.join(os.path.dirname(__file__), '..', 'ai', 'ia_marc', 'book', 'Cerebellum_Light.bin')
        if os.path.exists(book_path):
            engine.set_opening_book(book_path, "polyglot")
            engine.enable_opening_book(True)
            print(f"Livre d'ouvertures chargé: {book_path}")
        else:
            print(f"Livre d'ouvertures non trouvé: {book_path}")
            
        print(f"Moteur initialisé au niveau {level}")
        return True
    except Exception as e:
        print(f"Erreur initialisation moteur: {e}")
        return False


class ChessAPIHandler(SimpleHTTPRequestHandler):
    """Gestionnaire HTTP pour l'API d'échecs"""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        # API endpoints
        if parsed.path == '/api/levels':
            self.send_json_response(self.get_levels())
        elif parsed.path == '/api/personalities':
            self.send_json_response(self.get_personalities())
        elif parsed.path == '/api/status':
            self.send_json_response({'status': 'ok', 'engine': engine is not None})
        else:
            # Servir les fichiers statiques
            super().do_GET()
    
    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        
        if parsed.path == '/api/move':
            response = self.get_ai_move(data)
            self.send_json_response(response)
        elif parsed.path == '/api/set_level':
            response = self.set_level(data)
            self.send_json_response(response)
        elif parsed.path == '/api/reset':
            response = self.reset_engine()
            self.send_json_response(response)
        else:
            self.send_error(404, 'Not Found')
    
    def send_json_response(self, data):
        """Envoie une réponse JSON"""
        response = json.dumps(data)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def do_OPTIONS(self):
        """Gère les requêtes CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_levels(self):
        """Retourne les niveaux disponibles"""
        levels = []
        for key, level in DIFFICULTY_LEVELS.items():
            levels.append({
                'key': key,
                'name': level.name,
                'elo': level.elo,
                'depth': level.depth_limit,
                'description': f"ELO {level.elo}"
            })
        return {'levels': levels}
    
    def get_personalities(self):
        """Retourne les personnalités disponibles"""
        personalities = []
        for key, p in PERSONALITIES.items():
            personalities.append({
                'key': key,
                'name': p.name,
                'description': p.description
            })
        return {'personalities': personalities}
    
    def get_ai_move(self, data):
        """Obtient le meilleur coup de l'IA"""
        global engine
        
        if engine is None:
            init_engine(current_level)
        
        fen = data.get('fen', chess.STARTING_FEN)
        
        try:
            board = chess.Board(fen)
            
            # Obtenir le coup de l'IA
            move = engine.get_move(board)
            
            if move:
                # Jouer le coup pour obtenir le SAN
                san = board.san(move)
                
                return {
                    'success': True,
                    'move': {
                        'from': chess.square_name(move.from_square),
                        'to': chess.square_name(move.to_square),
                        'promotion': chess.piece_symbol(move.promotion).lower() if move.promotion else None,
                        'san': san,
                        'uci': move.uci()
                    },
                    'stats': engine.get_stats()
                }
            else:
                return {'success': False, 'error': 'Aucun coup trouvé'}
                
        except Exception as e:
            print(f"Erreur get_ai_move: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def set_level(self, data):
        """Change le niveau de difficulté"""
        global engine, current_level
        
        level = data.get('level', 'LEVEL5')
        
        try:
            if engine is None:
                init_engine(level)
            else:
                engine.set_level(level)
                current_level = level
            
            return {'success': True, 'level': level}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def reset_engine(self):
        """Réinitialise le moteur"""
        global engine
        
        try:
            if engine:
                engine.reset()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def log_message(self, format, *args):
        """Logging personnalisé"""
        if '/api/' in args[0]:
            print(f"[API] {args[0]}")


def run_server(port=8080):
    """Lance le serveur"""
    global engine
    
    # Changer le répertoire de travail
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialiser le moteur
    print("Initialisation de l'IA Marc V2...")
    if not init_engine():
        print("ATTENTION: Le moteur n'a pas pu être initialisé")
    
    # Démarrer le serveur
    server = HTTPServer(('', port), ChessAPIHandler)
    print(f"\n{'='*50}")
    print(f"Smart Chess - Serveur démarré")
    print(f"Interface: http://localhost:{port}")
    print(f"API: http://localhost:{port}/api/")
    print(f"{'='*50}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServeur arrêté")
        server.shutdown()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Smart Chess Server')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port du serveur')
    args = parser.parse_args()
    
    run_server(args.port)
