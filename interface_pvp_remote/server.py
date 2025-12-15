#!/usr/bin/env python3
"""
Interface PvP Remote - Serveur WebSocket
Serveur combinant HTTP (fichiers statiques) et WebSocket (temps réel).
"""

import asyncio
import json
import os
import sys
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import threading
import time

# Ajouter le chemin pour python-chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import websockets
except ImportError:
    print("Installation de websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

from hardware_bridge import HardwareBridge
from game_manager import GameManager, PlayerType


class PvPRemoteServer:
    """Serveur principal pour le jeu PvP à distance."""
    
    def __init__(self, http_port: int = 8081, ws_port: int = 8082, simulation: bool = False):
        self.http_port = http_port
        self.ws_port = ws_port
        self.simulation = simulation
        
        # Composants
        self.game = GameManager()
        self.hardware = HardwareBridge(simulation=simulation)
        
        # WebSocket clients
        self.clients: set = set()
        
        # État
        self.running = False
        
        # Configurer le callback pour les changements de capteurs
        self.hardware.set_state_change_callback(self._on_board_state_change)
    
    def _on_board_state_change(self, lifted, placed):
        """Callback appelé quand l'état du plateau change."""
        print(f"[Server] Changement détecté - lifted: {lifted}, placed: {placed}")
        
        # Vérifier que c'est le tour du joueur physique
        if self.game.get_current_player() != PlayerType.PHYSICAL:
            print("[Server] Ce n'est pas le tour du joueur physique, ignoré")
            return
        
        # Attendre la stabilisation
        time.sleep(0.5)
        
        # Détecter le coup
        move = self.game.detect_move_from_state_change(lifted, placed)
        
        if move:
            from_sq, to_sq = move
            print(f"[Server] Coup physique détecté: {from_sq} -> {to_sq}")
            
            # Valider et jouer le coup
            move_result = self.game.make_move(from_sq, to_sq)
            
            if move_result:
                # Illuminer le coup joué
                self.hardware.highlight_move(from_sq, to_sq)
                
                # Notifier tous les clients
                asyncio.run_coroutine_threadsafe(
                    self._broadcast({
                        "type": "physical_move",
                        "move": move_result,
                        "game_status": self.game.get_game_status()
                    }),
                    self.loop
                )
            else:
                print(f"[Server] Coup invalide: {from_sq} -> {to_sq}")
                asyncio.run_coroutine_threadsafe(
                    self._broadcast({
                        "type": "error",
                        "message": f"Coup illégal: {from_sq}-{to_sq}"
                    }),
                    self.loop
                )
    
    async def _broadcast(self, message: dict):
        """Envoie un message à tous les clients connectés."""
        if self.clients:
            msg = json.dumps(message)
            await asyncio.gather(
                *[client.send(msg) for client in self.clients],
                return_exceptions=True
            )
    
    async def _handle_websocket(self, websocket, path):
        """Gère une connexion WebSocket."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"[WS] Client connecté: {client_addr}")
        
        try:
            # Envoyer l'état initial
            await websocket.send(json.dumps({
                "type": "init",
                "game_status": self.game.get_game_status(),
                "simulation": self.simulation
            }))
            
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"[WS] Client déconnecté: {client_addr}")
        finally:
            self.clients.discard(websocket)
    
    async def _handle_message(self, websocket, message: str):
        """Traite un message reçu d'un client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "web_move":
                await self._handle_web_move(websocket, data)
            
            elif msg_type == "simulate_physical_move":
                await self._handle_simulate_physical_move(websocket, data)
            
            elif msg_type == "reset":
                await self._handle_reset(websocket)
            
            elif msg_type == "get_status":
                await websocket.send(json.dumps({
                    "type": "status",
                    "game_status": self.game.get_game_status()
                }))
            
            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
            
            else:
                print(f"[WS] Message inconnu: {msg_type}")
                
        except json.JSONDecodeError:
            print(f"[WS] Message invalide: {message}")
    
    async def _handle_web_move(self, websocket, data: dict):
        """Traite un coup du joueur web."""
        from_sq = data.get("from")
        to_sq = data.get("to")
        promotion = data.get("promotion")
        
        print(f"[Server] Coup web reçu: {from_sq} -> {to_sq}")
        
        # Vérifier que c'est le tour du joueur web
        if self.game.get_current_player() != PlayerType.WEB:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Ce n'est pas votre tour"
            }))
            return
        
        # Jouer le coup
        move_result = self.game.make_move(from_sq, to_sq, promotion)
        
        if move_result:
            # Illuminer le coup sur le plateau physique
            self.hardware.highlight_move(from_sq, to_sq)
            
            # Notifier tous les clients
            await self._broadcast({
                "type": "web_move",
                "move": move_result,
                "game_status": self.game.get_game_status()
            })
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Coup illégal: {from_sq}-{to_sq}"
            }))
    
    async def _handle_simulate_physical_move(self, websocket, data: dict):
        """Simule un coup du joueur physique (mode simulation uniquement)."""
        if not self.simulation:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Simulation désactivée"
            }))
            return
        
        from_sq = data.get("from")
        to_sq = data.get("to")
        
        print(f"[Server] Simulation coup physique: {from_sq} -> {to_sq}")
        
        # Vérifier que c'est le tour du joueur physique
        if self.game.get_current_player() != PlayerType.PHYSICAL:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Ce n'est pas le tour du joueur physique"
            }))
            return
        
        # Jouer le coup
        move_result = self.game.make_move(from_sq, to_sq)
        
        if move_result:
            # Simuler le déplacement sur le plateau
            self.hardware.simulate_move(from_sq, to_sq)
            self.hardware.highlight_move(from_sq, to_sq)
            
            # Notifier tous les clients
            await self._broadcast({
                "type": "physical_move",
                "move": move_result,
                "game_status": self.game.get_game_status()
            })
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Coup illégal: {from_sq}-{to_sq}"
            }))
    
    async def _handle_reset(self, websocket):
        """Réinitialise la partie."""
        print("[Server] Réinitialisation de la partie")
        self.game.reset()
        self.hardware.clear_all_leds()
        
        await self._broadcast({
            "type": "reset",
            "game_status": self.game.get_game_status()
        })
    
    def _run_http_server(self):
        """Lance le serveur HTTP dans un thread séparé."""
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        handler = SimpleHTTPRequestHandler
        httpd = HTTPServer(('', self.http_port), handler)
        print(f"[HTTP] Serveur démarré sur http://localhost:{self.http_port}")
        
        while self.running:
            httpd.handle_request()
    
    async def start(self):
        """Démarre le serveur."""
        self.running = True
        self.loop = asyncio.get_event_loop()
        
        # Démarrer le polling des capteurs
        self.hardware.start_polling(interval=0.1)
        
        # Démarrer le serveur HTTP dans un thread
        http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        http_thread.start()
        
        # Démarrer le serveur WebSocket
        print(f"[WS] Serveur démarré sur ws://localhost:{self.ws_port}")
        
        print("\n" + "=" * 60)
        print("Interface PvP Remote - Serveur démarré")
        print(f"  Interface Web: http://localhost:{self.http_port}")
        print(f"  WebSocket:     ws://localhost:{self.ws_port}")
        print(f"  Mode:          {'SIMULATION' if self.simulation else 'HARDWARE'}")
        print("=" * 60 + "\n")
        
        async with websockets.serve(self._handle_websocket, "", self.ws_port):
            await asyncio.Future()  # Run forever
    
    def stop(self):
        """Arrête le serveur."""
        self.running = False
        self.hardware.cleanup()
        print("[Server] Arrêt du serveur")


def main():
    parser = argparse.ArgumentParser(description='Interface PvP Remote Server')
    parser.add_argument('--http-port', type=int, default=8081, help='Port HTTP')
    parser.add_argument('--ws-port', type=int, default=8082, help='Port WebSocket')
    parser.add_argument('--simulation', '-s', action='store_true', help='Mode simulation')
    args = parser.parse_args()
    
    server = PvPRemoteServer(
        http_port=args.http_port,
        ws_port=args.ws_port,
        simulation=args.simulation
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n[Server] Interruption par l'utilisateur")
        server.stop()


if __name__ == "__main__":
    main()
