# SmartChess

## Project Overview

This repository contains the source code for SmartChess, an intelligent electronic chessboard. The project aims to build a physical chessboard that detects piece positions using reed sensors and provides visual feedback with LEDs. The software component includes a chess AI that can play against a human opponent.

The project is structured into two main parts:

-   **`prototypes/`**: Contains the hardware designs (KiCad files) and firmware for the physical chessboard prototypes. It includes a 2x2 proof of concept and an 8x8 mockup.
-   **`ai/`**: Contains the chess AI engine and related tools. The AI is written in Python and uses a variety of techniques, including alpha-beta pruning and neural networks.

The AI implementation is split between a custom bitboard-based chess engine (`ai/Chess.py`) for performance-critical parts and the `python-chess` library (`ai/engine.py`) for interfacing with UCI and XBoard protocols.

## Building and Running

The project is primarily written in Python. To run the AI, you'll need to install the dependencies listed in `prototypes/8x8-maquette/firmware/IA-Marc/V2/requirements.txt`.

### Dependencies

-   `python-chess>=1.999`
-   `psutil>=5.9.0`
-   `pytest>=7.0.0`
-   `pytest-benchmark>=4.0.0`

### Running the AI

The `ai` directory contains several scripts for running and testing the AI. The main entry point for the AI is likely `ai/engine.py`, which can be used to run a UCI or XBoard-compatible chess engine.

```bash
# TODO: Add specific commands for running the AI
# Example (hypothetical):
# python ai/engine.py --protocol uci
```

### Testing

The project uses `pytest` for testing. The `ai` directory contains several test scripts (e.g., `ai/Tests.py`, `ai/test_engines_v2.py`).

```bash
# TODO: Add specific commands for running tests
# Example (hypothetical):
# pytest ai/
```

## Development Conventions

-   The project uses a mix of a custom chess library and the `python-chess` library. The custom library in `ai/Chess.py` is used for the core board representation and move generation, likely for performance reasons. The `python-chess` library is used for higher-level tasks like protocol implementation (UCI/XBoard) and board serialization.
-   The `ai` directory contains a large number of files, many of which appear to be experiments and different versions of the AI. It would be beneficial to consolidate the code and remove unused files.
-   The project uses a virtual environment for managing dependencies, as indicated by the `venv` directories.
