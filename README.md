# SmartChess

An intelligent electronic chessboard that automatically detects piece positions and allows you to play against a powerful AI opponent.

## Project Structure

- `prototypes/`: Hardware designs (KiCad) and firmware for the physical chessboard.
  - `8x8-maquette/firmware/IA-Marc/V2/`: Contains the main application and the AI engine.
- `ai/`: Original research and development for the chess AI.
- `docs/`: Project documentation.

## Quick Start

This guide explains how to run the main chess application on a Raspberry Pi or a development machine.

### 1. Setup

First, set up a Python virtual environment and install the required dependencies.

```bash
# Navigate to the V2 AI directory
cd prototypes/8x8-maquette/firmware/IA-Marc/V2/

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Game

The main script allows you to play against the AI. It includes a menu to select the difficulty level.

```bash
# From the V2 directory, run the menu-driven game script
python3 chess_game_IA_menu.py
```

### AI Difficulty Levels

The AI has 9 difficulty levels, ranging from ELO 200 (Novice) to 1800 (Expert). You can select the level through the on-screen menu at the start of the game.

### Opening Book (Optional, for stronger AI)

For an even stronger AI, especially in the opening phase, you can download a Polyglot book file.
Download `Cerebellum_Light.bin` (or a similar `.bin` book) from:
[https://zipproth.de/Brainfish/download/](https://zipproth.de/Brainfish/download/)

Place the downloaded `.bin` file into the `prototypes/8x8-maquette/firmware/IA-Marc/book/` directory.
