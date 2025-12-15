/**
 * Smart Chess PvP Remote - Application Logic
 * Interface de jeu à distance avec plateau physique
 */

// Configuration
const WS_PORT = 8082;
const WS_URL = `ws://${window.location.hostname || 'localhost'}:${WS_PORT}`;

// État global
const state = {
    ws: null,
    connected: false,
    simulation: false,
    game: null,
    selectedSquare: null,
    validMoves: [],
    lastMove: null,
    isMyTurn: true,
    moveHistory: []
};

// Symboles des pièces
const PIECE_SYMBOLS = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
};

// ========== Initialisation ==========

document.addEventListener('DOMContentLoaded', () => {
    initializeGame();
    setupEventListeners();
});

function initializeGame() {
    state.game = new Chess();
    renderBoard();
}

function setupEventListeners() {
    document.getElementById('btn-connect').addEventListener('click', connect);
    document.getElementById('btn-reset').addEventListener('click', resetGame);

    document.getElementById('btn-new-game').addEventListener('click', () => {
        showScreen('game-screen');
        resetGame();
    });

    document.getElementById('btn-disconnect').addEventListener('click', () => {
        if (state.ws) state.ws.close();
        showScreen('connect-screen');
    });

    document.getElementById('btn-sim-move').addEventListener('click', simulatePhysicalMove);

    document.getElementById('sim-from').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') document.getElementById('sim-to').focus();
    });
    document.getElementById('sim-to').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') simulatePhysicalMove();
    });
}

// ========== WebSocket ==========

function connect() {
    console.log('Connexion à', WS_URL);
    updateConnectionStatus('connecting');

    try {
        state.ws = new WebSocket(WS_URL);

        state.ws.onopen = () => {
            console.log('WebSocket connecté');
            state.connected = true;
            updateConnectionStatus('connected');
            showScreen('game-screen');
            showToast('Connecté au serveur', 'success');
        };

        state.ws.onclose = () => {
            console.log('WebSocket déconnecté');
            state.connected = false;
            updateConnectionStatus('disconnected');
        };

        state.ws.onerror = (error) => {
            console.error('WebSocket erreur:', error);
            updateConnectionStatus('error');
            showToast('Impossible de se connecter au serveur', 'error');
        };

        state.ws.onmessage = handleMessage;

    } catch (error) {
        console.error('Erreur de connexion:', error);
        updateConnectionStatus('error');
        showToast('Erreur de connexion', 'error');
    }
}

function handleMessage(event) {
    try {
        const data = JSON.parse(event.data);
        console.log('Message reçu:', data);

        switch (data.type) {
            case 'init':
                handleInit(data);
                break;
            case 'web_move':
                handleWebMove(data);
                break;
            case 'physical_move':
                handlePhysicalMove(data);
                break;
            case 'reset':
                handleReset(data);
                break;
            case 'error':
                showToast(data.message, 'error');
                break;
            case 'status':
                updateGameStatus(data.game_status);
                break;
        }
    } catch (error) {
        console.error('Erreur parsing message:', error);
    }
}

function sendMessage(data) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(data));
    }
}

// ========== Gestionnaires de messages ==========

function handleInit(data) {
    state.simulation = data.simulation;

    if (state.simulation) {
        document.getElementById('simulation-panel').classList.remove('hidden');
        document.getElementById('simulation-note').style.display = 'block';
    }

    updateGameStatus(data.game_status);
}

function handleWebMove(data) {
    applyMove(data.move);
    updateGameStatus(data.game_status);

    // Afficher l'indicateur LED
    showLedIndicator(data.move.from, data.move.to);

    showToast('Coup envoyé au plateau : ' + data.move.san, 'success');
}

function handlePhysicalMove(data) {
    applyMove(data.move);
    updateGameStatus(data.game_status);

    // Cacher l'indicateur LED
    hideLedIndicator();

    showToast('Coup reçu du plateau : ' + data.move.san);
}

function handleReset(data) {
    state.game = new Chess();
    state.selectedSquare = null;
    state.lastMove = null;
    state.moveHistory = [];
    hideLedIndicator();
    renderBoard();
    updateGameStatus(data.game_status);
    showToast('Nouvelle partie démarrée');
}

// ========== LED Indicator ==========

function showLedIndicator(from, to) {
    const indicator = document.getElementById('led-indicator');
    const squares = document.getElementById('led-squares');
    squares.textContent = `${from} → ${to}`;
    indicator.style.display = 'flex';
}

function hideLedIndicator() {
    document.getElementById('led-indicator').style.display = 'none';
}

// ========== Actions ==========

function makeMove(from, to, promotion = null) {
    if (!state.connected || !state.isMyTurn) return;

    sendMessage({
        type: 'web_move',
        from: from,
        to: to,
        promotion: promotion
    });
}

function simulatePhysicalMove() {
    const fromInput = document.getElementById('sim-from');
    const toInput = document.getElementById('sim-to');

    const from = fromInput.value.toLowerCase().trim();
    const to = toInput.value.toLowerCase().trim();

    if (from.length !== 2 || to.length !== 2) {
        showToast('Format invalide (ex: e7, e5)', 'error');
        return;
    }

    sendMessage({
        type: 'simulate_physical_move',
        from: from,
        to: to
    });

    fromInput.value = '';
    toInput.value = '';
    fromInput.focus();
}

function resetGame() {
    sendMessage({ type: 'reset' });
}

// ========== Rendu de l'échiquier ==========

function renderBoard() {
    const board = document.getElementById('chessboard');
    board.innerHTML = '';

    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const file = String.fromCharCode(97 + col);
            const rank = 8 - row;
            const squareName = file + rank;

            const square = document.createElement('div');
            square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
            square.dataset.square = squareName;

            square.addEventListener('click', () => onSquareClick(squareName));

            board.appendChild(square);
        }
    }

    updateBoard();
}

function updateBoard() {
    if (!state.game) return;

    const squares = document.querySelectorAll('.square');
    const position = state.game.board();

    squares.forEach(square => {
        const squareName = square.dataset.square;
        const col = squareName.charCodeAt(0) - 97;
        const row = 8 - parseInt(squareName[1]);
        const piece = position[row][col];

        square.classList.remove('selected', 'valid-move', 'valid-capture', 'last-move', 'check');

        if (piece) {
            const symbol = piece.color === 'w'
                ? PIECE_SYMBOLS[piece.type.toUpperCase()]
                : PIECE_SYMBOLS[piece.type];

            const isDisabled = !state.isMyTurn || piece.color !== 'w';
            square.innerHTML = `<span class="piece ${isDisabled ? 'disabled' : ''}" data-color="${piece.color}">${symbol}</span>`;
        } else {
            square.innerHTML = '';
        }

        if (state.lastMove) {
            if (squareName === state.lastMove.from || squareName === state.lastMove.to) {
                square.classList.add('last-move');
            }
        }
    });

    if (state.game.in_check()) {
        const turn = state.game.turn();
        const kingSquare = findKingSquare(turn);
        if (kingSquare) {
            const el = document.querySelector(`[data-square="${kingSquare}"]`);
            if (el) el.classList.add('check');
        }
    }
}

function findKingSquare(color) {
    const board = state.game.board();
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const piece = board[row][col];
            if (piece && piece.type === 'k' && piece.color === color) {
                return String.fromCharCode(97 + col) + (8 - row);
            }
        }
    }
    return null;
}

// ========== Interaction ==========

function onSquareClick(squareName) {
    if (!state.isMyTurn) {
        showToast("En attente du coup adverse...", "error");
        return;
    }

    const piece = state.game.get(squareName);

    if (state.selectedSquare) {
        if (state.validMoves.includes(squareName)) {
            const move = attemptMove(state.selectedSquare, squareName);
            if (move) {
                makeMove(state.selectedSquare, squareName, move.promotion);
            }
        }
        deselectSquare();
    } else if (piece && piece.color === 'w') {
        selectSquare(squareName);
    }
}

function selectSquare(squareName) {
    deselectSquare();
    state.selectedSquare = squareName;

    const el = document.querySelector(`[data-square="${squareName}"]`);
    if (el) el.classList.add('selected');

    const moves = state.game.moves({ square: squareName, verbose: true });
    state.validMoves = moves.map(m => m.to);

    moves.forEach(move => {
        const sq = document.querySelector(`[data-square="${move.to}"]`);
        if (sq) {
            sq.classList.add(move.captured ? 'valid-capture' : 'valid-move');
        }
    });
}

function deselectSquare() {
    if (state.selectedSquare) {
        const el = document.querySelector(`[data-square="${state.selectedSquare}"]`);
        if (el) el.classList.remove('selected');
    }

    document.querySelectorAll('.valid-move, .valid-capture').forEach(sq => {
        sq.classList.remove('valid-move', 'valid-capture');
    });

    state.selectedSquare = null;
    state.validMoves = [];
}

function attemptMove(from, to) {
    const piece = state.game.get(from);
    const toRank = parseInt(to[1]);

    if (piece && piece.type === 'p') {
        if ((piece.color === 'w' && toRank === 8) || (piece.color === 'b' && toRank === 1)) {
            return { from, to, promotion: 'q' };
        }
    }

    return { from, to, promotion: null };
}

function applyMove(moveData) {
    const move = state.game.move({
        from: moveData.from,
        to: moveData.to,
        promotion: moveData.promotion || undefined
    });

    if (move) {
        state.lastMove = { from: moveData.from, to: moveData.to };
        state.moveHistory.push(moveData);
        updateBoard();
        updateMovesDisplay();
    }
}

// ========== UI Updates ==========

function updateGameStatus(status) {
    if (!status) return;

    if (status.fen && state.game.fen() !== status.fen) {
        state.game.load(status.fen);
        updateBoard();
    }

    state.isMyTurn = status.current_player === 'web';

    const whiteBar = document.getElementById('player-white');
    const blackBar = document.getElementById('player-black');
    const whiteStatus = document.getElementById('status-white');
    const blackStatus = document.getElementById('status-black');

    whiteBar.classList.remove('active', 'waiting');
    blackBar.classList.remove('active', 'waiting');

    if (state.isMyTurn) {
        whiteBar.classList.add('active');
        blackBar.classList.add('waiting');
        whiteStatus.style.display = 'inline';
        blackStatus.style.display = 'none';
    } else {
        blackBar.classList.add('active');
        whiteBar.classList.add('waiting');
        blackStatus.style.display = 'inline';
        whiteStatus.style.display = 'none';
    }

    const statusEl = document.getElementById('status');
    if (status.is_checkmate) {
        statusEl.textContent = 'Échec et mat !';
        showEndScreen(status);
    } else if (status.is_stalemate) {
        statusEl.textContent = 'Pat - Match nul';
        showEndScreen(status);
    } else if (status.is_check) {
        statusEl.textContent = state.isMyTurn ? '⚠️ Échec ! Votre tour' : '⚠️ Échec ! Attente du plateau...';
    } else {
        statusEl.textContent = state.isMyTurn ? '🎯 Votre tour - Jouez un coup' : '⏳ En attente du coup sur le plateau...';
    }

    if (status.last_move) {
        state.lastMove = { from: status.last_move.from, to: status.last_move.to };
    }
}

function updateMovesDisplay() {
    const movesEl = document.getElementById('moves');
    const pgn = [];

    for (let i = 0; i < state.moveHistory.length; i += 2) {
        const moveNum = Math.floor(i / 2) + 1;
        let moveStr = `${moveNum}.${state.moveHistory[i].san}`;
        if (state.moveHistory[i + 1]) {
            moveStr += ` ${state.moveHistory[i + 1].san}`;
        }
        pgn.push(moveStr);
    }

    movesEl.textContent = pgn.slice(-4).join(' ');
}

function updateConnectionStatus(status) {
    const container = document.getElementById('connection-status');
    const dot = container.querySelector('.status-dot');
    const text = container.querySelector('span:last-child');

    dot.className = 'status-dot';

    switch (status) {
        case 'connected':
            text.textContent = 'Connecté';
            break;
        case 'connecting':
            dot.classList.add('connecting');
            text.textContent = 'Connexion...';
            break;
        case 'disconnected':
        case 'error':
            dot.classList.add('disconnected');
            text.textContent = 'Déconnecté';
            break;
    }
}

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.add('hidden'));
    document.getElementById(screenId).classList.remove('hidden');
}

function showEndScreen(status) {
    const title = document.getElementById('end-title');
    const subtitle = document.getElementById('end-subtitle');

    if (status.result === 'white') {
        title.textContent = '🎉 Victoire !';
        subtitle.textContent = 'Vous avez battu le joueur physique';
    } else if (status.result === 'black') {
        title.textContent = '😔 Défaite';
        subtitle.textContent = 'Le joueur physique a gagné';
    } else {
        title.textContent = '🤝 Match nul';
        subtitle.textContent = 'La partie est nulle';
    }

    document.getElementById('stat-moves').textContent = status.move_count;
    showScreen('end-screen');
}

function showToast(message, type = '') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast' + (type ? ` ${type}` : '');
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}
