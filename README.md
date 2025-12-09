# SmartChess: Intelligent Electronic Chessboard

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Project Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-c51a4a.svg)]()

**An intelligent electronic chessboard that automatically detects piece positions using reed sensors and lets you play against a powerful embedded AI engine.**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Hardware Design](#hardware-design)
- [Vision System](#vision-system)
- [AI Engine](#ai-engine)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

SmartChess is a complete smart chessboard solution combining custom hardware design with a powerful embedded chess AI. The system features:

- **Real-time piece detection** via a 64-reed sensor matrix (one per square)
- **Computer vision backup** using CNN detection and optical flow tracking for error verification
- **Visual feedback** through a 64-LED matrix indicating valid moves, threats, and game state
- **Embedded AI** optimized for Raspberry Pi 5, supporting 8 difficulty levels (400-2400 ELO)
- **Custom PCB design** with KiCad schematics for the complete 8×8 board

<div align="center">
<img src="docs/img/rendu_final_coffrage1.jpg" alt="SmartChess Final Assembly" width="350"/>
<img src="docs/img/rendu_final_pieces.jpg" alt="SmartChess with Pieces" width="350"/>
<br>
<em>Left: Final wooden enclosure | Right: Board with chess pieces</em>
</div>

---

## Key Features

### Hardware

- **64 Reed Sensors**: Magnetic detection using 4× MCP23017 I/O expanders
- **64+8 LED Matrix**: Visual feedback via 2× HT16K33 LED drivers
- **TCA9548A Multiplexer**: Centralized I²C bus management
- **Custom PCB**: Complete KiCad schematic for the 8×8 design

### Software

- **IA-Marc V2 Engine**: Optimized chess engine for embedded systems
- **50K-200K nodes/second** on Raspberry Pi 5
- **8 Difficulty Levels**: From beginner (400 ELO) to expert (2400 ELO)
- **6 Personalities**: Aggressive, Defensive, Positional, Tactical, Materialist, Balanced
- **Opening Book**: Polyglot format support for natural openings

### Vision System

- **CNN-based Detection**: Neural network for chessboard corner localization
- **Lucas-Kanade Tracking**: Optical flow for real-time corner tracking
- **Reed Sensor Fusion**: Cross-validation between vision and magnetic sensors
- **Error Detection**: Automatic discrepancy detection for move verification

### AI Engine Optimizations

| Optimization | Speedup | ELO Gain |
|-------------|---------|----------|
| Transposition Table | 3-5× | +200 |
| Null Move Pruning | 1.5-2× | +100 |
| Lazy SMP (4 threads) | 2.5-3× | +100 |
| Late Move Reduction | 1.5× | +80 |
| Killer Moves | 1.3× | +50 |
| **Total Cumulative** | **30-180×** | **+650 ELO** |

---


## Hardware Design

<div align="center">
<img src="docs/img/modelisation3D.png" alt="3D Model" width="600"/>
<br>
<em>3D CAD model of the SmartChess board</em>
</div>

### Component List

| Component | Quantity | Role |
|-----------|----------|------|
| Raspberry Pi 5 (8GB) | 1 | Main processor |
| TCA9548A | 1 | I²C multiplexer hub |
| MCP23017 | 4 | 16-pin I/O controllers for sensors |
| Reed Sensors | 64 | Magnetic piece detection |
| HT16K33 | 2 | LED matrix drivers |
| LEDs | 72 (64+8) | Visual feedback |

### I²C Bus Configuration

| Channel | Component | Address | Function |
|---------|-----------|---------|----------|
| 0 | MCP23017 (CM0) | 0x20 | Rows 1-2 sensors |
| 1 | MCP23017 (CM1) | 0x20 | Rows 3-4 sensors |
| 2 | MCP23017 (CM2) | 0x20 | Rows 5-6 sensors |
| 3 | MCP23017 (CM3) | 0x20 | Rows 7-8 sensors |
| 4 | HT16K33 (LED_A) | 0x70 | 8×8 LED matrix |
| 5 | HT16K33 (LED_B) | 0x71 | Extra 1×8 LED row |
| 6 | Camera (USB/CSI) | - | Vision system input |

---

## Vision System

The vision subsystem provides a secondary detection layer to complement reed sensors, enabling piece tracking, move verification, and error detection.

### Detection Pipeline

| Stage | Method | Purpose |
|-------|--------|----------|
| **Initial Detection** | CNN (PyTorch) | Locate 4 chessboard corners with high accuracy |
| **Real-time Tracking** | Lucas-Kanade Optical Flow | Track corners at 30+ FPS with minimal latency |
| **State Management** | Finite State Machine | Switch between detection/tracking based on confidence |
| **Sensor Fusion** | VisionReedBridge | Compare vision output with reed sensor readings |

### Key Features

**CNN Corner Detection:**
- Lightweight neural network optimized for Raspberry Pi
- Detects 4 board corners regardless of perspective
- Automatic re-detection when tracking confidence drops

**Lucas-Kanade Tracking:**
- 30+ FPS real-time corner tracking
- Pyramidal implementation for robustness
- Sub-pixel accuracy for precise square mapping

**Vision-Reed Fusion:**
- Cross-validates piece positions between sensors
- Detects sensor malfunctions or cheating attempts
- Provides confidence scores for each detected state

### Usage

```python
from vision.chessboard_detector import ChessboardDetector, create_detector
from vision.integration import VisionReedBridge

# Create detector with Raspberry Pi preset
detector = create_detector("raspberry_pi")
detector.load_model("models/corner_detector.pt")

# Process camera frame
result = detector.process_frame(frame)
if result.detected:
    corners = result.corners  # 4 corner positions

# Compare with reed sensors
bridge = VisionReedBridge(corners)
bridge.update_reed_state(reed_matrix, timestamp)
bridge.update_vision_state(vision_squares, timestamp, result.confidence)

comparison = bridge.compare_states()
if not comparison.matches:
    print(f"Discrepancies detected: {comparison.discrepancies}")
```

---

## AI Engine

### Difficulty Levels

| Level | ELO | Depth | Time | Error Rate | Description |
|-------|-----|-------|------|------------|-------------|
| Enfant | 400 | 1 | 0.3s | 40% | Simple moves, many mistakes |
| Débutant | 600 | 2 | 0.5s | 30% | Plays superficially |
| Amateur | 1000 | 3 | 1.0s | 20% | Understands basics |
| Club | 1400 | 4 | 2.0s | 10% | Good club player |
| Compétition | 1800 | 6 | 4.0s | 5% | Regional competition level |
| Expert | 2000 | 8 | 8.0s | 2% | Expert with minor flaws |
| Maître | 2200 | 10 | 15s | 0% | FIDE master level |
| Maximum | 2400 | 20 | 30s | 0% | Maximum RPi 5 power |

### Engine Features

**Search Algorithms:**
- NegaMax with Alpha-Beta pruning
- Iterative Deepening with Aspiration Windows
- Quiescence Search for capture stability
- Null Move Pruning for aggressive cutoffs
- Late Move Reduction (LMR)

**Evaluation:**
- PeSTO evaluation tables
- Mobility analysis
- Pawn structure analysis
- King safety evaluation
- Piece coordination scoring

**Performance:**
- Transposition Table (256-512 MB)
- Killer Moves heuristic
- History Heuristic
- Lazy SMP parallelization (4 threads)
- PyPy compatible for 2-3× speedup

---

## Project Structure

```
smart-chess/
├── README.md                    # Project overview (this file)
├── LICENSE                      # MIT License
│
├── docs/                        # Documentation and images
│   └── img/                     # Project images
│       ├── modelisation3D.png
│       ├── rendu_final_coffrage1.jpg
│       └── rendu_final_pieces.jpg
│
├── ai/                          # Original AI R&D
│   ├── Chess.py                 # Core chess logic
│   ├── engine.py                # Search engines
│   ├── evaluator.py             # Position evaluation
│   └── ...
│
├── ia_marc/                     # Production AI engines
│   ├── V1/                      # Version 1 (legacy)
│   ├── V2/                      # Version 2 (current)
│   │   ├── engine_main.py       # Main API
│   │   ├── engine_brain.py      # PeSTO evaluation
│   │   ├── engine_search.py     # NegaMax search
│   │   ├── engine_tt.py         # Transposition table
│   │   ├── engine_ordering.py   # Move ordering
│   │   ├── engine_opening.py    # Opening book
│   │   ├── engine_config.py     # Configuration
│   │   ├── requirements.txt     # Dependencies
│   │   └── tests/               # Test suite
│   └── book/                    # Opening books (Polyglot)
│
└── prototypes/                  # Hardware prototypes
    ├── echiquier_8x8/           # Main 8×8 prototype
    │   ├── firmware/            # Embedded code
    │   │   ├── ia_embarquee/    # Game scripts
    │   │   │   ├── chess_game_v1.py
    │   │   │   └── chess_game_v2.py
    │   │   ├── vision/          # Vision system
    │   │   │   ├── chessboard_detector.py  # Main detector
    │   │   │   ├── integration.py          # Reed sensor bridge
    │   │   │   ├── detection/              # CNN model & preprocessing
    │   │   │   └── tracking/               # LK tracker & state mgmt
    │   │   └── requirements.txt
    │   └── hardware/            # KiCad schematics
    │       ├── 8x8.kicad_sch
    │       └── 8x8.pdf
    └── echiquier_2x2/           # Test prototype (2×2)
```

---

## Getting Started

### Prerequisites

- Raspberry Pi 5 (8GB recommended) with Raspberry Pi OS 64-bit
- Python 3.10+ or PyPy3 for maximum performance
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/promaaa/smart-chess.git
cd smart-chess

# Navigate to the AI engine directory
cd ia_marc/V2/

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Game

```bash
# Navigate to the game scripts
cd prototypes/echiquier_8x8/firmware/ia_embarquee/

# Run the main game (V2 with menu)
python3 chess_game_v2.py
```

### Quick AI Test

```python
from ia_marc.V2.engine_main import ChessEngine
import chess

# Create engine
engine = ChessEngine()

# Set difficulty
engine.set_level("Club")  # or engine.set_elo(1400)

# Get a move
board = chess.Board()
move = engine.get_move(board, time_limit=3.0)

print(f"Best move: {move}")
```

### Opening Book (Optional)

For stronger openings, download a Polyglot opening book:

```bash
# Create book directory
mkdir -p ia_marc/book/

# Download a book (example: Cerebellum Light)
# Place the .bin file in ia_marc/book/
```

---

## Documentation

### Technical References

| Document | Location | Description |
|----------|----------|-------------|
| Hardware Structure | `prototypes/echiquier_8x8/firmware/structure.md` | Full hardware documentation |
| AI Engine README | `ia_marc/V2/README.md` | Detailed engine documentation |
| Hardware Schematic | `prototypes/echiquier_8x8/hardware/8x8.pdf` | KiCad schematic export |

### Key APIs

**ChessEngine (ia_marc/V2/engine_main.py)**:
- `get_move(board, time_limit)`: Get best move for position
- `set_level(name)`: Set difficulty by name
- `set_elo(elo)`: Set difficulty by ELO (400-2400)
- `set_personality(name)`: Set playing style
- `get_move_with_stats(board)`: Get move with search statistics

### Future Improvements

- [ ] Camera-based piece recognition (computer vision)
- [ ] UCI protocol support for external GUI
- [ ] Web interface for remote play
- [ ] Neural network evaluation (planned)
- [ ] Endgame tablebases support

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution

- Performance optimization of the search algorithm
- Adding tactical test positions
- Extending the opening book
- Improving position evaluation heuristics
- Hardware design improvements

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [python-chess](https://python-chess.readthedocs.io/) - Chess library for Python
- [PeSTO](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function) - Piece-Square tables
- [Chess Programming Wiki](https://www.chessprogramming.org/) - Invaluable resource for chess programming techniques
- Adafruit libraries for hardware interfacing (HT16K33, MCP23017, TCA9548A)
