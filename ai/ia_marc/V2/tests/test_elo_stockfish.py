
import chess
import chess.engine
import subprocess
import sys
import time
import os
import logging
from typing import Optional, Dict

# Add the parent directory to the sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine_main import ChessEngine
from engine_config import DIFFICULTY_LEVELS

# --- Configuration ---
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Replace with your Stockfish path
ENGINE_UCI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chess_engine_uci.py'))
NUM_GAMES_PER_LEVEL = 2 # Play an even number of games for color balance

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper to start/stop engine ---
async def run_match(
    ai_level: str,
    stockfish_elo: int,
    num_games: int,
    initial_fen: str = chess.STARTING_FEN
) -> Dict[str, int]:
    """
    Runs a series of games between IA-Marc V2 and Stockfish.
    """
    logger.info(f"--- Starting match: IA-Marc V2 ({ai_level}) vs Stockfish ({stockfish_elo} ELO) ---")

    ai_engine_name = f"IA-Marc V2 ({ai_level})"
    stockfish_engine_name = f"Stockfish ({stockfish_elo} ELO)"

    results = {
        ai_engine_name: 0,
        stockfish_engine_name: 0,
        "Draws": 0
    }

    ai_engine_instance = None
    ai_process = None
    stockfish_engine_instance = None

    try:
        # Start AI engine
        transport, protocol = await chess.engine.popen_uci([sys.executable, ENGINE_UCI_PATH])
        logger.info(f"Engine {ENGINE_UCI_PATH} started.")
        logger.info(f"Transport: {transport}")
        logger.info(f"Protocol: {protocol}")
        ai_engine_instance = protocol
        
        # Configure IA-Marc V2 level
        await ai_engine_instance.set_option("Level", ai_level)
        await ai_engine_instance.is_ready()
        logger.info(f"IA-Marc V2 set to level: {ai_level}")

        # Start Stockfish engine
        stockfish_engine_instance = await chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        await stockfish_engine_instance.setoption("UCI_LimitStrength", True)
        await stockfish_engine_instance.setoption("UCI_Elo", stockfish_elo)
        await stockfish_engine_instance.setoption("Threads", 1) # Ensure consistent behavior for ELO setting
        await stockfish_engine_instance.setoption("Hash", 128)
        logger.info(f"Stockfish set to {stockfish_elo} ELO.")

        for i in range(num_games):
            board = chess.Board(initial_fen)
            game_result = "1/2-1/2" # Default to draw

            # Alternate colors
            if i % 2 == 0:
                white_player = ai_engine_instance
                black_player = stockfish_engine_instance
                logger.info(f"Game {i+1}: IA-Marc V2 (White) vs Stockfish (Black)")
            else:
                white_player = stockfish_engine_instance
                black_player = ai_engine_instance
                logger.info(f"Game {i+1}: Stockfish (White) vs IA-Marc V2 (Black)")
            
            # Reset engines for new game
            await white_player.ucinewgame()
            await black_player.ucinewgame()
            await white_player.isready()
            await black_player.isready()


            while not board.is_game_over():
                try:
                    if board.turn == chess.WHITE:
                        player_to_move = white_player
                    else:
                        player_to_move = black_player
                    
                    # Time limit based on AI's configured level for fairness
                    # Using the time_limit from the AI's config, scaled down
                    time_limit_per_move = DIFFICULTY_LEVELS[ai_level].time_limit * 0.5 
                    if time_limit_per_move < 0.1: time_limit_per_move = 0.1 # Minimum time

                    result = await player_to_move.play(board, chess.engine.Limit(time=time_limit_per_move))
                    board.push(result.move)
                    logger.debug(f"Game {i+1} - Move: {result.move.uci()} - FEN: {board.fen()}")

                except chess.engine.EngineError as e:
                    logger.error(f"Engine error in game {i+1}: {e}")
                    # Assume loss for the engine that errors out
                    if player_to_move == ai_engine_instance:
                        game_result = "0-1" if i % 2 == 0 else "1-0" # AI was white and lost, or AI was black and won (Stockfish's perspective)
                    else:
                        game_result = "1-0" if i % 2 == 0 else "0-1" # Stockfish was white and lost, or Stockfish was black and won (AI's perspective)
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in game {i+1}: {e}")
                    # Assume loss for the engine that errors out
                    if player_to_move == ai_engine_instance:
                        game_result = "0-1" if i % 2 == 0 else "1-0"
                    else:
                        game_result = "1-0" if i % 2 == 0 else "0-1"
                    break

            if board.is_game_over():
                game_result = board.result()
            
            if game_result == "1-0":
                if i % 2 == 0: # AI was white and won
                    results[ai_engine_name] += 1
                else: # Stockfish was white and won
                    results[stockfish_engine_name] += 1
            elif game_result == "0-1":
                if i % 2 == 0: # AI was white and lost
                    results[stockfish_engine_name] += 1
                else: # Stockfish was white and lost
                    results[ai_engine_name] += 1
            else: # Draw
                results["Draws"] += 1
            
            logger.info(f"Game {i+1} finished. Result: {game_result}")

    except Exception as e:
        logger.error(f"An error occurred during the match: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if stockfish_engine_instance:
            await stockfish_engine_instance.quit()
        if ai_engine_instance:
            await ai_engine_instance.quit()
            # If for some reason ai_engine_instance.quit() doesn't terminate the process
            if ai_process and ai_process.returncode is None:
                ai_process.terminate()
                await ai_process.wait()

    logger.info(f"--- Match results: IA-Marc V2 ({ai_level}) vs Stockfish ({stockfish_elo} ELO) ---")
    for player, score in results.items():
        logger.info(f"{player}: {score}")
    logger.info("-" * 60)

    return results

# --- Main execution ---
import asyncio

async def main():
    # Identify the best AI level from engine_config.py (now LEVEL12)
    ai_level_to_test = "LEVEL12" 
    ai_level_details = DIFFICULTY_LEVELS[ai_level_to_test]
    
    logger.info(f"--- Starting ELO evaluation for {ai_level_details.name} (target ELO: {ai_level_details.elo}) ---")

    # ELO levels for Stockfish to test against
    stockfish_elo_levels = range(1800, 2601, 200)

    for stockfish_elo in stockfish_elo_levels:
        results = await run_match(
            ai_level=ai_level_to_test,
            stockfish_elo=stockfish_elo,
            num_games=NUM_GAMES_PER_LEVEL
        )

        ia_marc_wins = results.get(f"IA-Marc V2 ({ai_level_to_test})", 0)
        stockfish_wins = results.get(f"Stockfish ({stockfish_elo} ELO)", 0)
        draws = results.get("Draws", 0)

        logger.info(f"\n--- Overall Results for IA-Marc V2 ({ai_level_to_test}) vs Stockfish ({stockfish_elo} ELO) ---")
        logger.info(f"IA-Marc V2 Wins: {ia_marc_wins}")
        logger.info(f"Stockfish Wins: {stockfish_wins}")
        logger.info(f"Draws: {draws}")

        # Simple logic to estimate ELO
        if ia_marc_wins > stockfish_wins:
            logger.info(f"IA-Marc V2 is likely stronger than {stockfish_elo} ELO.")
        elif stockfish_wins > ia_marc_wins:
            logger.info(f"IA-Marc V2 is likely weaker than {stockfish_elo} ELO. Stopping evaluation here.")
            break
        else:
            logger.info(f"IA-Marc V2 is roughly equivalent to {stockfish_elo} ELO.")

    logger.info("--- ELO Evaluation Finished ---")

if __name__ == "__main__":
    asyncio.run(main())
