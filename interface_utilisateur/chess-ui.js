/**
 * Smart Chess - Chess UI Module
 * Gestion de l'interface utilisateur de l'échiquier
 */

const ChessUI = (function() {
    'use strict';

    // Symboles Unicode des pièces
    const PIECE_SYMBOLS = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
    };

    // Valeurs des pièces pour le calcul du matériel
    const PIECE_VALUES = {
        'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9,
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9
    };

    // Configuration
    const config = {
        animationDuration: 200,
        showValidMoves: true,
        showLastMove: true,
        soundEnabled: true
    };

    // État de sélection
    let selectedSquare = null;
    let validMoves = [];
    let lastMove = { from: null, to: null };
    let isFlipped = false;

    // Références DOM
    let boardElement = null;
    let game = null;

    /**
     * Initialise l'échiquier avec le jeu Chess.js
     */
    function init(chessGame, boardElementId = 'chessboard') {
        game = chessGame;
        boardElement = document.getElementById(boardElementId);
        
        if (!boardElement) {
            console.error('Board element not found:', boardElementId);
            return;
        }

        renderBoard();
        addEventListeners();
    }

    /**
     * Rendu complet de l'échiquier
     */
    function renderBoard() {
        if (!boardElement) return;
        
        boardElement.innerHTML = '';
        
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const actualRow = isFlipped ? 7 - row : row;
                const actualCol = isFlipped ? 7 - col : col;
                
                const file = String.fromCharCode(97 + actualCol); // a-h
                const rank = 8 - actualRow; // 8-1
                const squareName = file + rank;
                
                const square = createSquare(squareName, actualRow, actualCol);
                boardElement.appendChild(square);
            }
        }
        
        updateBoard();
    }

    /**
     * Crée un élément de case
     */
    function createSquare(squareName, row, col) {
        const square = document.createElement('div');
        square.className = 'square';
        square.dataset.square = squareName;
        
        // Couleur de la case (alternance)
        if ((row + col) % 2 === 0) {
            square.classList.add('light');
        } else {
            square.classList.add('dark');
        }
        
        return square;
    }

    /**
     * Met à jour l'échiquier avec la position actuelle
     */
    function updateBoard() {
        if (!game || !boardElement) return;

        const squares = boardElement.querySelectorAll('.square');
        
        squares.forEach(square => {
            const squareName = square.dataset.square;
            const piece = game.get(squareName);
            
            // Réinitialiser les classes
            square.classList.remove('selected', 'valid-move', 'valid-capture', 'last-move', 'check');
            
            // Afficher la pièce
            if (piece) {
                const symbol = getPieceSymbol(piece);
                square.innerHTML = `<span class="piece" data-color="${piece.color}">${symbol}</span>`;
            } else {
                square.innerHTML = '';
            }
        });

        // Mettre en évidence le dernier coup
        if (config.showLastMove && lastMove.from && lastMove.to) {
            highlightLastMove();
        }

        // Mettre en évidence l'échec
        highlightCheck();
    }

    /**
     * Obtient le symbole Unicode d'une pièce
     */
    function getPieceSymbol(piece) {
        const key = piece.color === 'w' ? piece.type.toUpperCase() : piece.type.toLowerCase();
        return PIECE_SYMBOLS[key] || '';
    }

    /**
     * Ajoute les écouteurs d'événements
     */
    function addEventListeners() {
        boardElement.addEventListener('click', handleSquareClick);
        
        // Raccourcis clavier
        document.addEventListener('keydown', handleKeyboard);
    }

    /**
     * Gère le clic sur une case
     */
    function handleSquareClick(event) {
        const square = event.target.closest('.square');
        if (!square) return;

        const squareName = square.dataset.square;
        const piece = game.get(squareName);

        // Si aucune case n'est sélectionnée
        if (!selectedSquare) {
            // Sélectionner si c'est une pièce du joueur actif
            if (piece && piece.color === game.turn()) {
                selectSquare(squareName);
            }
            return;
        }

        // Si on clique sur la même case, désélectionner
        if (selectedSquare === squareName) {
            deselectSquare();
            return;
        }

        // Si on clique sur une autre pièce de la même couleur
        if (piece && piece.color === game.turn()) {
            selectSquare(squareName);
            return;
        }

        // Tenter de jouer le coup
        const move = attemptMove(selectedSquare, squareName);
        
        if (move) {
            // Coup valide, déclencher l'événement
            const event = new CustomEvent('playerMove', { detail: { move } });
            document.dispatchEvent(event);
        }

        deselectSquare();
    }

    /**
     * Sélectionne une case et affiche les coups valides
     */
    function selectSquare(squareName) {
        deselectSquare();
        
        selectedSquare = squareName;
        const squareElement = boardElement.querySelector(`[data-square="${squareName}"]`);
        
        if (squareElement) {
            squareElement.classList.add('selected');
        }

        // Calculer et afficher les coups valides
        if (config.showValidMoves) {
            showValidMoves(squareName);
        }
    }

    /**
     * Désélectionne la case actuelle
     */
    function deselectSquare() {
        if (selectedSquare) {
            const squareElement = boardElement.querySelector(`[data-square="${selectedSquare}"]`);
            if (squareElement) {
                squareElement.classList.remove('selected');
            }
        }
        
        selectedSquare = null;
        hideValidMoves();
    }

    /**
     * Affiche les coups valides pour une case
     */
    function showValidMoves(fromSquare) {
        validMoves = game.moves({ square: fromSquare, verbose: true });
        
        validMoves.forEach(move => {
            const squareElement = boardElement.querySelector(`[data-square="${move.to}"]`);
            if (squareElement) {
                if (move.captured) {
                    squareElement.classList.add('valid-capture');
                } else {
                    squareElement.classList.add('valid-move');
                }
            }
        });
    }

    /**
     * Cache les coups valides
     */
    function hideValidMoves() {
        const squares = boardElement.querySelectorAll('.valid-move, .valid-capture');
        squares.forEach(square => {
            square.classList.remove('valid-move', 'valid-capture');
        });
        validMoves = [];
    }

    /**
     * Tente de jouer un coup
     */
    function attemptMove(from, to) {
        // Vérifier si c'est un coup de promotion
        const piece = game.get(from);
        const isPromotion = piece && piece.type === 'p' && 
                           ((piece.color === 'w' && to[1] === '8') || 
                            (piece.color === 'b' && to[1] === '1'));

        if (isPromotion) {
            // Afficher le dialogue de promotion
            return new Promise((resolve) => {
                showPromotionDialog(piece.color, (promotionPiece) => {
                    const move = game.move({ from, to, promotion: promotionPiece });
                    if (move) {
                        lastMove = { from, to };
                        updateBoard();
                        resolve(move);
                    } else {
                        resolve(null);
                    }
                });
            });
        }

        // Coup normal
        const move = game.move({ from, to });
        
        if (move) {
            lastMove = { from, to };
            updateBoard();
            return move;
        }
        
        return null;
    }

    /**
     * Affiche le dialogue de promotion
     */
    function showPromotionDialog(color, callback) {
        const pieces = color === 'w' ? ['Q', 'R', 'B', 'N'] : ['q', 'r', 'b', 'n'];
        
        // Créer l'overlay
        const overlay = document.createElement('div');
        overlay.className = 'overlay';
        
        // Créer le dialogue
        const dialog = document.createElement('div');
        dialog.className = 'promotion-dialog';
        dialog.innerHTML = `
            <h3>Choisir la promotion</h3>
            <div class="promotion-options">
                ${pieces.map(p => `
                    <button class="promotion-piece" data-piece="${p.toLowerCase()}">
                        ${PIECE_SYMBOLS[p]}
                    </button>
                `).join('')}
            </div>
        `;

        // Gérer les clics
        dialog.querySelectorAll('.promotion-piece').forEach(btn => {
            btn.addEventListener('click', () => {
                const piece = btn.dataset.piece;
                document.body.removeChild(overlay);
                document.body.removeChild(dialog);
                callback(piece);
            });
        });

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
    }

    /**
     * Met en évidence le dernier coup
     */
    function highlightLastMove() {
        if (lastMove.from) {
            const fromSquare = boardElement.querySelector(`[data-square="${lastMove.from}"]`);
            if (fromSquare) fromSquare.classList.add('last-move');
        }
        if (lastMove.to) {
            const toSquare = boardElement.querySelector(`[data-square="${lastMove.to}"]`);
            if (toSquare) toSquare.classList.add('last-move');
        }
    }

    /**
     * Met en évidence l'échec
     */
    function highlightCheck() {
        if (game.in_check()) {
            // Trouver le roi en échec
            const turn = game.turn();
            const board = game.board();
            
            for (let row = 0; row < 8; row++) {
                for (let col = 0; col < 8; col++) {
                    const piece = board[row][col];
                    if (piece && piece.type === 'k' && piece.color === turn) {
                        const file = String.fromCharCode(97 + col);
                        const rank = 8 - row;
                        const squareName = file + rank;
                        const squareElement = boardElement.querySelector(`[data-square="${squareName}"]`);
                        if (squareElement) {
                            squareElement.classList.add('check');
                        }
                        return;
                    }
                }
            }
        }
    }

    /**
     * Joue un coup de l'IA (avec animation)
     */
    function playAIMove(moveNotation) {
        return new Promise((resolve) => {
            const move = game.move(moveNotation);
            
            if (move) {
                lastMove = { from: move.from, to: move.to };
                
                // Animation simple
                setTimeout(() => {
                    updateBoard();
                    resolve(move);
                }, config.animationDuration);
            } else {
                resolve(null);
            }
        });
    }

    /**
     * Annule le dernier coup
     */
    function undoMove() {
        const move = game.undo();
        if (move) {
            // Annuler aussi le coup de l'IA si c'était au tour du joueur
            if (game.turn() !== 'w') {
                game.undo();
            }
            lastMove = { from: null, to: null };
            updateBoard();
        }
        return move;
    }

    /**
     * Réinitialise l'échiquier
     */
    function reset() {
        game.reset();
        selectedSquare = null;
        validMoves = [];
        lastMove = { from: null, to: null };
        updateBoard();
    }

    /**
     * Retourne l'échiquier (vue noire)
     */
    function flipBoard() {
        isFlipped = !isFlipped;
        renderBoard();
    }

    /**
     * Obtient les pièces capturées
     */
    function getCapturedPieces() {
        const initialPieces = {
            w: { p: 8, n: 2, b: 2, r: 2, q: 1 },
            b: { p: 8, n: 2, b: 2, r: 2, q: 1 }
        };

        const currentPieces = { w: {}, b: {} };
        const board = game.board();

        // Compter les pièces actuelles
        board.forEach(row => {
            row.forEach(piece => {
                if (piece) {
                    const color = piece.color;
                    const type = piece.type;
                    currentPieces[color][type] = (currentPieces[color][type] || 0) + 1;
                }
            });
        });

        // Calculer les pièces capturées
        const captured = { w: [], b: [] };
        
        ['w', 'b'].forEach(color => {
            const oppositeColor = color === 'w' ? 'b' : 'w';
            Object.keys(initialPieces[color]).forEach(type => {
                const initial = initialPieces[color][type];
                const current = currentPieces[color][type] || 0;
                const capturedCount = initial - current;
                
                for (let i = 0; i < capturedCount; i++) {
                    captured[oppositeColor].push(PIECE_SYMBOLS[color === 'w' ? type.toUpperCase() : type]);
                }
            });
        });

        return captured;
    }

    /**
     * Gestion du clavier
     */
    function handleKeyboard(event) {
        switch (event.key.toLowerCase()) {
            case 'escape':
                deselectSquare();
                break;
            case 'f':
                flipBoard();
                break;
        }
    }

    /**
     * Met à jour la configuration
     */
    function setConfig(newConfig) {
        Object.assign(config, newConfig);
    }

    // API publique
    return {
        init,
        updateBoard,
        playAIMove,
        undoMove,
        reset,
        flipBoard,
        getCapturedPieces,
        setConfig,
        getLastMove: () => lastMove,
        isSquareSelected: () => selectedSquare !== null
    };
})();

// Export pour utilisation dans d'autres modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChessUI;
}
