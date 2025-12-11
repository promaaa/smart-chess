/**
 * Smart Chess - Application
 */

(function () {
    'use strict';

    // État
    const state = {
        game: null,
        level: 'LEVEL5',
        personality: 'EQUILIBRE',
        playerTurn: true,
        moveCount: 0,
        moves: [],
        started: false
    };

    // DOM
    const $ = id => document.getElementById(id);

    async function init() {
        state.game = new Chess();

        // Init backend
        if (AIEngine.init) await AIEngine.init();

        // Render
        await renderLevels();
        renderPersonalities();
        ChessUI.init(state.game);

        // Events
        $('btn-start').onclick = startGame;
        $('btn-undo').onclick = undo;
        $('btn-menu').onclick = () => showScreen('menu');
        $('btn-replay').onclick = startGame;
        $('btn-back').onclick = () => showScreen('menu');
        document.addEventListener('playerMove', onPlayerMove);

        console.log('Smart Chess ready');
    }

    async function renderLevels() {
        const levels = await AIEngine.getLevels();
        $('levels-grid').innerHTML = levels.map(l => {
            const num = l.key.replace('LEVEL', '');
            const name = l.name.includes('(') ? l.name.split('(')[1].replace(')', '') : l.name;
            return `<div class="level-card ${l.key === state.level ? 'selected' : ''}" data-level="${l.key}">
                <div class="level-num">Niveau ${num}</div>
                <div class="level-name">${name}</div>
                <div class="level-elo">ELO ${l.elo}</div>
            </div>`;
        }).join('');

        $('levels-grid').querySelectorAll('.level-card').forEach(card => {
            card.onclick = () => {
                $('levels-grid').querySelectorAll('.level-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                state.level = card.dataset.level;
            };
        });
    }

    function renderPersonalities() {
        const list = AIEngine.getPersonalities();
        $('personalities').innerHTML = list.map(p =>
            `<div class="personality ${p.key === state.personality ? 'selected' : ''}" data-key="${p.key}">${p.name}</div>`
        ).join('');

        $('personalities').querySelectorAll('.personality').forEach(el => {
            el.onclick = () => {
                $('personalities').querySelectorAll('.personality').forEach(e => e.classList.remove('selected'));
                el.classList.add('selected');
                state.personality = el.dataset.key;
            };
        });
    }

    function startGame() {
        state.game.reset();
        state.playerTurn = true;
        state.moveCount = 0;
        state.moves = [];
        state.started = true;

        AIEngine.setLevel(state.level);
        AIEngine.setPersonality(state.personality);

        const level = AIEngine.getCurrentLevel();
        $('ai-level').textContent = level.name;
        $('captured-white').textContent = '';
        $('captured-black').textContent = '';
        $('moves').textContent = '';

        ChessUI.reset();
        showScreen('game');
        updateStatus('Votre tour');
    }

    async function onPlayerMove(e) {
        if (!state.started || !state.playerTurn) return;

        const move = await e.detail.move;
        if (!move) return;

        state.moveCount++;
        state.moves.push(move.san);
        updateMoves();
        updateCaptured();

        if (checkEnd()) return;

        // AI turn
        state.playerTurn = false;
        updateStatus('IA Marc réfléchit...');

        setTimeout(async () => {
            const aiMove = await AIEngine.getBestMove(state.game);
            if (aiMove) {
                await ChessUI.playAIMove(aiMove);
                state.moveCount++;
                state.moves.push(aiMove.san || aiMove.from + aiMove.to);
                updateMoves();
                updateCaptured();

                if (checkEnd()) return;

                state.playerTurn = true;
                updateStatus('Votre tour');
            }
        }, 200);
    }

    function checkEnd() {
        if (!state.game.game_over()) {
            if (state.game.in_check()) {
                updateStatus(state.game.turn() === 'w' ? 'Échec !' : 'IA en échec !');
            }
            return false;
        }

        state.started = false;
        let title, subtitle;

        if (state.game.in_checkmate()) {
            if (state.game.turn() === 'w') {
                title = 'Défaite';
                subtitle = 'Échec et mat';
            } else {
                title = 'Victoire !';
                subtitle = 'Vous avez gagné';
            }
        } else {
            title = 'Nulle';
            subtitle = 'Partie nulle';
        }

        $('end-title').textContent = title;
        $('end-subtitle').textContent = subtitle;
        $('stat-moves').textContent = state.moveCount;
        showScreen('end');
        return true;
    }

    function undo() {
        if (!state.started || !state.playerTurn || state.moves.length < 2) return;
        ChessUI.undoMove();
        state.moves.pop();
        state.moves.pop();
        state.moveCount = Math.max(0, state.moveCount - 2);
        updateMoves();
        updateCaptured();
        updateStatus('Votre tour');
    }

    function showScreen(name) {
        ['menu', 'game', 'end'].forEach(s => {
            $(s + '-screen').classList.toggle('hidden', s !== name);
        });
    }

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    function updateMoves() {
        $('moves').textContent = state.moves.slice(-6).join(' ');
    }

    function updateCaptured() {
        const c = ChessUI.getCapturedPieces();
        $('captured-white').textContent = c.b.join('');
        $('captured-black').textContent = c.w.join('');
    }

    document.addEventListener('DOMContentLoaded', init);
})();
