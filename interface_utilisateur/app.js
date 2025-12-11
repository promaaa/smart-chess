/**
 * Smart Chess - Application Principale
 * Gestion de l'application et coordination des modules
 * Version sans emojis
 */

(function () {
    'use strict';

    // État de l'application
    const state = {
        game: null,
        selectedLevel: 'LEVEL5',
        selectedPersonality: 'EQUILIBRE',
        isPlayerTurn: true,
        moveCount: 0,
        captureCount: 0,
        moveHistory: [],
        gameStarted: false
    };

    // Références DOM
    const elements = {
        levelSelectionScreen: null,
        gameScreen: null,
        gameOverScreen: null,
        levelsGrid: null,
        personalitySelector: null,
        btnStartGame: null,
        btnUndo: null,
        btnResign: null,
        btnNewGame: null,
        btnPlayAgain: null,
        btnChangeLevel: null,
        gameStatus: null,
        moveHistory: null,
        aiLevelDisplay: null,
        capturedByAi: null,
        capturedByPlayer: null,
        ledStrip: null,
        gameOverIcon: null,
        gameOverTitle: null,
        gameOverSubtitle: null,
        statMoves: null,
        statCaptures: null
    };

    /**
     * Initialise l'application
     */
    function init() {
        // Initialiser le jeu Chess.js
        state.game = new Chess();

        // Récupérer les références DOM
        cacheElements();

        // Construire l'interface de sélection
        renderLevelSelection();
        renderPersonalitySelection();

        // Initialiser l'échiquier
        ChessUI.init(state.game);

        // Attacher les événements
        attachEventListeners();

        // Animation d'entrée
        animateLEDs([1, 2, 3, 4, 5, 6, 7, 8], 'sequence');

        console.log('Smart Chess: Application initialisee');
    }

    /**
     * Cache les références DOM
     */
    function cacheElements() {
        elements.levelSelectionScreen = document.getElementById('level-selection-screen');
        elements.gameScreen = document.getElementById('game-screen');
        elements.gameOverScreen = document.getElementById('game-over-screen');
        elements.levelsGrid = document.getElementById('levels-grid');
        elements.personalitySelector = document.getElementById('personality-selector');
        elements.btnStartGame = document.getElementById('btn-start-game');
        elements.btnUndo = document.getElementById('btn-undo');
        elements.btnResign = document.getElementById('btn-resign');
        elements.btnNewGame = document.getElementById('btn-new-game');
        elements.btnPlayAgain = document.getElementById('btn-play-again');
        elements.btnChangeLevel = document.getElementById('btn-change-level');
        elements.gameStatus = document.getElementById('game-status');
        elements.moveHistory = document.getElementById('move-history');
        elements.aiLevelDisplay = document.getElementById('ai-level-display');
        elements.capturedByAi = document.getElementById('captured-by-ai');
        elements.capturedByPlayer = document.getElementById('captured-by-player');
        elements.ledStrip = document.getElementById('led-strip');
        elements.gameOverIcon = document.getElementById('game-over-icon');
        elements.gameOverTitle = document.getElementById('game-over-title');
        elements.gameOverSubtitle = document.getElementById('game-over-subtitle');
        elements.statMoves = document.getElementById('stat-moves');
        elements.statCaptures = document.getElementById('stat-captures');
    }

    /**
     * Génère l'interface de sélection des niveaux
     */
    function renderLevelSelection() {
        const levels = AIEngine.getLevels();

        elements.levelsGrid.innerHTML = levels.map(level => {
            const levelNum = level.key.replace('LEVEL', '');
            const levelName = level.name.split('(')[1] ? level.name.split('(')[1].replace(')', '') : level.name;
            return '<div class="level-card ' + (level.key === state.selectedLevel ? 'selected' : '') + '" data-level="' + level.key + '">' +
                '<span class="level-number">Niveau ' + levelNum + '</span>' +
                '<span class="level-name">' + levelName + '</span>' +
                '<span class="level-elo">ELO ' + level.elo + '</span>' +
                '</div>';
        }).join('');

        // Événements de sélection
        elements.levelsGrid.querySelectorAll('.level-card').forEach(function (card) {
            card.addEventListener('click', function () {
                elements.levelsGrid.querySelectorAll('.level-card').forEach(function (c) {
                    c.classList.remove('selected');
                });
                card.classList.add('selected');
                state.selectedLevel = card.dataset.level;
                updateLEDIndicator(parseInt(card.dataset.level.replace('LEVEL', '')));
            });
        });
    }

    /**
     * Génère l'interface de sélection des personnalités
     */
    function renderPersonalitySelection() {
        const personalities = AIEngine.getPersonalities();

        elements.personalitySelector.innerHTML = personalities.map(function (p) {
            return '<div class="personality-chip ' + (p.key === state.selectedPersonality ? 'selected' : '') + '" data-personality="' + p.key + '">' +
                p.name +
                '</div>';
        }).join('');

        // Événements de sélection
        elements.personalitySelector.querySelectorAll('.personality-chip').forEach(function (chip) {
            chip.addEventListener('click', function () {
                elements.personalitySelector.querySelectorAll('.personality-chip').forEach(function (c) {
                    c.classList.remove('selected');
                });
                chip.classList.add('selected');
                state.selectedPersonality = chip.dataset.personality;
            });
        });
    }

    /**
     * Attache tous les événements
     */
    function attachEventListeners() {
        // Bouton démarrer la partie
        elements.btnStartGame.addEventListener('click', startGame);

        // Boutons d'action en jeu
        elements.btnUndo.addEventListener('click', undoMove);
        elements.btnResign.addEventListener('click', resignGame);
        elements.btnNewGame.addEventListener('click', function () {
            showScreen('levelSelection');
        });

        // Boutons fin de partie
        elements.btnPlayAgain.addEventListener('click', startGame);
        elements.btnChangeLevel.addEventListener('click', function () {
            showScreen('levelSelection');
        });

        // Événement de coup joué
        document.addEventListener('playerMove', handlePlayerMove);

        // Raccourcis clavier
        document.addEventListener('keydown', handleKeyboard);
    }

    /**
     * Démarre une nouvelle partie
     */
    function startGame() {
        // Réinitialiser l'état
        state.game.reset();
        state.isPlayerTurn = true;
        state.moveCount = 0;
        state.captureCount = 0;
        state.moveHistory = [];
        state.gameStarted = true;

        // Configurer l'IA
        AIEngine.setLevel(state.selectedLevel);
        AIEngine.setPersonality(state.selectedPersonality);

        // Mettre à jour l'affichage
        var level = AIEngine.getCurrentLevel();
        elements.aiLevelDisplay.textContent = level.name;
        elements.capturedByAi.innerHTML = '';
        elements.capturedByPlayer.innerHTML = '';
        elements.moveHistory.textContent = '';

        // Réinitialiser l'échiquier
        ChessUI.reset();

        // Afficher l'écran de jeu
        showScreen('game');
        updateStatus('Votre tour - Jouez les blancs');

        // Animation des LEDs
        animateLEDs([1, 2, 3, 4, 5, 6, 7, 8], 'flash');

        console.log('Partie demarree: ' + level.name + ', ' + AIEngine.getCurrentPersonality().name);
    }

    /**
     * Gère le coup du joueur
     */
    async function handlePlayerMove(event) {
        if (!state.gameStarted || !state.isPlayerTurn) return;

        var moveResult = await event.detail.move;
        if (!moveResult) return;

        state.moveCount++;
        if (moveResult.captured) state.captureCount++;

        // Ajouter à l'historique
        state.moveHistory.push(moveResult.san);
        updateMoveHistory();
        updateCapturedPieces();

        // Vérifier fin de partie
        if (checkGameOver()) return;

        // Tour de l'IA
        state.isPlayerTurn = false;
        updateStatus('IA Marc reflechit...');
        elements.gameStatus.classList.add('thinking');
        animateLEDs([3, 4, 5, 6], 'pulse');

        // Attendre et jouer le coup de l'IA
        setTimeout(async function () {
            await playAIMove();
        }, 300);
    }

    /**
     * Joue le coup de l'IA
     */
    async function playAIMove() {
        try {
            var aiMove = await AIEngine.getBestMove(state.game);

            if (aiMove) {
                await ChessUI.playAIMove(aiMove);

                state.moveCount++;
                if (aiMove.captured) state.captureCount++;

                state.moveHistory.push(aiMove.san || (aiMove.from + aiMove.to));
                updateMoveHistory();
                updateCapturedPieces();

                // Vérifier fin de partie
                if (checkGameOver()) return;

                // Retour au joueur
                state.isPlayerTurn = true;
                elements.gameStatus.classList.remove('thinking');
                updateStatus('Votre tour');
                animateLEDs([1], 'on');
            }
        } catch (error) {
            console.error('Erreur IA:', error);
            updateStatus('Erreur de l\'IA - Votre tour');
            state.isPlayerTurn = true;
            elements.gameStatus.classList.remove('thinking');
        }
    }

    /**
     * Vérifie si la partie est terminée
     */
    function checkGameOver() {
        if (state.game.game_over()) {
            var icon, title, subtitle;

            if (state.game.in_checkmate()) {
                if (state.game.turn() === 'w') {
                    // Les blancs ont perdu (joueur)
                    icon = '—';
                    title = 'Defaite';
                    subtitle = 'L\'IA Marc vous a mis echec et mat';
                    animateLEDs([1, 2, 3, 4, 5, 6, 7, 8], 'error');
                } else {
                    // Les noirs ont perdu (IA)
                    icon = '+';
                    title = 'Victoire !';
                    subtitle = 'Felicitations, vous avez battu l\'IA Marc';
                    animateLEDs([1, 2, 3, 4, 5, 6, 7, 8], 'celebration');
                }
            } else if (state.game.in_stalemate()) {
                icon = '=';
                title = 'Pat';
                subtitle = 'La partie est nulle par pat';
                animateLEDs([2, 4, 6, 8], 'warning');
            } else if (state.game.in_draw()) {
                icon = '=';
                title = 'Nulle';
                subtitle = 'La partie est nulle';
                animateLEDs([2, 4, 6, 8], 'warning');
            } else if (state.game.in_threefold_repetition()) {
                icon = '=';
                title = 'Nulle';
                subtitle = 'Nulle par repetition';
                animateLEDs([2, 4, 6, 8], 'warning');
            } else {
                icon = '*';
                title = 'Partie terminee';
                subtitle = 'La partie est terminee';
            }

            showGameOver(icon, title, subtitle);
            return true;
        }

        // Vérifier l'échec
        if (state.game.in_check()) {
            var who = state.game.turn() === 'w' ? 'Vous etes' : 'L\'IA est';
            updateStatus(who + ' en echec !');
            animateLEDs([4, 5], 'warning');
        }

        return false;
    }

    /**
     * Annule le dernier coup
     */
    function undoMove() {
        if (!state.gameStarted || !state.isPlayerTurn) return;
        if (state.moveHistory.length < 2) return;

        // Annuler deux coups (joueur + IA)
        ChessUI.undoMove();
        state.moveHistory.pop();
        state.moveHistory.pop();
        state.moveCount = Math.max(0, state.moveCount - 2);

        updateMoveHistory();
        updateCapturedPieces();
        updateStatus('Votre tour');
    }

    /**
     * Abandonne la partie
     */
    function resignGame() {
        if (!state.gameStarted) return;

        if (confirm('Etes-vous sur de vouloir abandonner ?')) {
            showGameOver('X', 'Abandon', 'Vous avez abandonne la partie');
            animateLEDs([1, 2, 3, 4, 5, 6, 7, 8], 'error');
        }
    }

    /**
     * Affiche l'écran de fin de partie
     */
    function showGameOver(icon, title, subtitle) {
        state.gameStarted = false;

        elements.gameOverIcon.textContent = icon;
        elements.gameOverTitle.textContent = title;
        elements.gameOverSubtitle.textContent = subtitle;
        elements.statMoves.textContent = state.moveCount;
        elements.statCaptures.textContent = state.captureCount;

        showScreen('gameOver');
    }

    /**
     * Affiche un écran spécifique
     */
    function showScreen(screenName) {
        elements.levelSelectionScreen.classList.add('hidden');
        elements.gameScreen.classList.add('hidden');
        elements.gameOverScreen.classList.add('hidden');

        switch (screenName) {
            case 'levelSelection':
                elements.levelSelectionScreen.classList.remove('hidden');
                break;
            case 'game':
                elements.gameScreen.classList.remove('hidden');
                break;
            case 'gameOver':
                elements.gameOverScreen.classList.remove('hidden');
                break;
        }
    }

    /**
     * Met à jour le message de statut
     */
    function updateStatus(message) {
        elements.gameStatus.textContent = message;
    }

    /**
     * Met à jour l'historique des coups
     */
    function updateMoveHistory() {
        var lastMoves = state.moveHistory.slice(-6);
        elements.moveHistory.textContent = lastMoves.join(' ');
    }

    /**
     * Met à jour les pièces capturées
     */
    function updateCapturedPieces() {
        var captured = ChessUI.getCapturedPieces();
        elements.capturedByAi.innerHTML = captured.b.join('');
        elements.capturedByPlayer.innerHTML = captured.w.join('');
    }

    /**
     * Animation des LEDs
     */
    function animateLEDs(leds, type) {
        var allLeds = elements.ledStrip.querySelectorAll('.led');

        // Réinitialiser toutes les LEDs
        allLeds.forEach(function (led) {
            led.classList.remove('active', 'warning', 'error');
        });

        switch (type) {
            case 'on':
                leds.forEach(function (num) {
                    var led = elements.ledStrip.querySelector('[data-led="' + num + '"]');
                    if (led) led.classList.add('active');
                });
                break;

            case 'sequence':
                leds.forEach(function (num, index) {
                    setTimeout(function () {
                        var led = elements.ledStrip.querySelector('[data-led="' + num + '"]');
                        if (led) led.classList.add('active');
                    }, index * 100);
                });
                break;

            case 'flash':
                leds.forEach(function (num) {
                    var led = elements.ledStrip.querySelector('[data-led="' + num + '"]');
                    if (led) {
                        led.classList.add('active');
                        setTimeout(function () { led.classList.remove('active'); }, 200);
                        setTimeout(function () { led.classList.add('active'); }, 400);
                    }
                });
                break;

            case 'pulse':
                var pulseOn = true;
                var pulseInterval = setInterval(function () {
                    leds.forEach(function (num) {
                        var led = elements.ledStrip.querySelector('[data-led="' + num + '"]');
                        if (led) {
                            if (pulseOn) {
                                led.classList.add('active');
                            } else {
                                led.classList.remove('active');
                            }
                        }
                    });
                    pulseOn = !pulseOn;
                }, 300);

                // Stopper après quelques secondes
                setTimeout(function () { clearInterval(pulseInterval); }, 3000);
                break;

            case 'warning':
                leds.forEach(function (num) {
                    var led = elements.ledStrip.querySelector('[data-led="' + num + '"]');
                    if (led) led.classList.add('warning');
                });
                break;

            case 'error':
                leds.forEach(function (num) {
                    var led = elements.ledStrip.querySelector('[data-led="' + num + '"]');
                    if (led) led.classList.add('error');
                });
                break;

            case 'celebration':
                var celebrationStep = 0;
                var celebrationInterval = setInterval(function () {
                    allLeds.forEach(function (led, i) {
                        led.classList.remove('active', 'warning');
                        if ((i + celebrationStep) % 2 === 0) {
                            led.classList.add('active');
                        }
                    });
                    celebrationStep++;
                    if (celebrationStep > 10) {
                        clearInterval(celebrationInterval);
                        allLeds.forEach(function (led) { led.classList.add('active'); });
                    }
                }, 200);
                break;
        }
    }

    /**
     * Met à jour l'indicateur LED pour le niveau sélectionné
     */
    function updateLEDIndicator(level) {
        var allLeds = elements.ledStrip.querySelectorAll('.led');
        allLeds.forEach(function (led, i) {
            led.classList.remove('active');
            if (i < level) {
                led.classList.add('active');
            }
        });
    }

    /**
     * Gestion du clavier
     */
    function handleKeyboard(event) {
        switch (event.key.toLowerCase()) {
            case 'n':
                if (!state.gameStarted) {
                    startGame();
                } else {
                    showScreen('levelSelection');
                }
                break;
            case 'u':
                undoMove();
                break;
            case 'escape':
                if (state.gameStarted) {
                    if (confirm('Retourner au menu ?')) {
                        showScreen('levelSelection');
                    }
                }
                break;
        }
    }

    // Initialiser l'application quand le DOM est prêt
    document.addEventListener('DOMContentLoaded', init);
})();
