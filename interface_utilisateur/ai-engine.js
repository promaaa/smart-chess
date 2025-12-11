/**
 * Smart Chess - AI Engine Module
 * Simulation de l'IA Marc V2 pour l'interface web
 * Version optimisée pour éviter les blocages
 */

const AIEngine = (function () {
    'use strict';

    // Configuration des niveaux (basée sur ia_marc/V2/engine_config.py)
    const DIFFICULTY_LEVELS = {
        'LEVEL1': {
            name: 'Niveau 1 (Débutant)',
            elo: 400,
            depth: 1,
            maxNodes: 500,
            timeLimit: 300,
            errorRate: 0.7,
            randomMoveChance: 0.4,
            description: 'Idéal pour les enfants et débutants'
        },
        'LEVEL2': {
            name: 'Niveau 2 (Novice)',
            elo: 600,
            depth: 2,
            maxNodes: 1000,
            timeLimit: 500,
            errorRate: 0.5,
            randomMoveChance: 0.25,
            description: 'Pour ceux qui apprennent les bases'
        },
        'LEVEL3': {
            name: 'Niveau 3 (Occasionnel)',
            elo: 800,
            depth: 2,
            maxNodes: 2000,
            timeLimit: 800,
            errorRate: 0.3,
            randomMoveChance: 0.15,
            description: 'Joueur occasionnel'
        },
        'LEVEL4': {
            name: 'Niveau 4 (Amateur)',
            elo: 1000,
            depth: 3,
            maxNodes: 5000,
            timeLimit: 1200,
            errorRate: 0.2,
            randomMoveChance: 0.08,
            description: 'Joueur amateur régulier'
        },
        'LEVEL5': {
            name: 'Niveau 5 (Club débutant)',
            elo: 1200,
            depth: 3,
            maxNodes: 8000,
            timeLimit: 1500,
            errorRate: 0.1,
            randomMoveChance: 0.03,
            description: 'Niveau club débutant'
        },
        'LEVEL6': {
            name: 'Niveau 6 (Club intermédiaire)',
            elo: 1400,
            depth: 3,
            maxNodes: 10000,
            timeLimit: 2000,
            errorRate: 0.05,
            randomMoveChance: 0.0,
            description: 'Niveau club intermédiaire'
        },
        'LEVEL7': {
            name: 'Niveau 7 (Club avancé)',
            elo: 1600,
            depth: 4,
            maxNodes: 15000,
            timeLimit: 2500,
            errorRate: 0.02,
            randomMoveChance: 0.0,
            description: 'Niveau club avancé'
        },
        'LEVEL8': {
            name: 'Niveau 8 (Maximum)',
            elo: 1700,
            depth: 4,
            maxNodes: 20000,
            timeLimit: 3000,
            errorRate: 0.0,
            randomMoveChance: 0.0,
            description: 'Force maximale de l\'IA Marc'
        }
    };

    // Personnalités de jeu
    const PERSONALITIES = {
        'EQUILIBRE': {
            name: 'Équilibré',
            description: 'Joue de manière équilibrée',
            bonuses: { attack: 0, defense: 0, center: 10, mobility: 5 }
        },
        'AGRESSIF': {
            name: 'Agressif',
            description: 'Attaque constamment',
            bonuses: { attack: 50, defense: -20, center: 15, mobility: 20 }
        },
        'DEFENSIF': {
            name: 'Défensif',
            description: 'Joue solidement',
            bonuses: { attack: -20, defense: 50, center: 5, mobility: 0 }
        },
        'POSITIONNEL': {
            name: 'Positionnel',
            description: 'Contrôle de l\'espace',
            bonuses: { attack: 0, defense: 10, center: 30, mobility: 25 }
        },
        'TACTIQUE': {
            name: 'Tactique',
            description: 'Cherche les combinaisons',
            bonuses: { attack: 40, defense: 0, center: 10, mobility: 30 }
        },
        'MATERIALISTE': {
            name: 'Matérialiste',
            description: 'Privilégie le matériel',
            bonuses: { attack: 10, defense: 10, center: 5, mobility: 5, material: 50 }
        }
    };

    // Valeurs des pièces pour l'évaluation
    const PIECE_VALUES = {
        'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000
    };

    // Tables de positionnement simplifiées
    const PST = {
        p: [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5, 5, 10, 25, 25, 10, 5, 5],
            [0, 0, 0, 20, 20, 0, 0, 0],
            [5, -5, -10, 0, 0, -10, -5, 5],
            [5, 10, 10, -20, -20, 10, 10, 5],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ],
        n: [
            [-50, -40, -30, -30, -30, -30, -40, -50],
            [-40, -20, 0, 0, 0, 0, -20, -40],
            [-30, 0, 10, 15, 15, 10, 0, -30],
            [-30, 5, 15, 20, 20, 15, 5, -30],
            [-30, 0, 15, 20, 20, 15, 0, -30],
            [-30, 5, 10, 15, 15, 10, 5, -30],
            [-40, -20, 0, 5, 5, 0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50]
        ],
        b: [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-10, 0, 5, 10, 10, 5, 0, -10],
            [-10, 5, 5, 10, 10, 5, 5, -10],
            [-10, 0, 10, 10, 10, 10, 0, -10],
            [-10, 10, 10, 10, 10, 10, 10, -10],
            [-10, 5, 0, 0, 0, 0, 5, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ],
        r: [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 10, 10, 10, 10, 10, 10, 5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [0, 0, 0, 5, 5, 0, 0, 0]
        ],
        q: [
            [-20, -10, -10, -5, -5, -10, -10, -20],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-10, 0, 5, 5, 5, 5, 0, -10],
            [-5, 0, 5, 5, 5, 5, 0, -5],
            [0, 0, 5, 5, 5, 5, 0, -5],
            [-10, 5, 5, 5, 5, 5, 0, -10],
            [-10, 0, 5, 0, 0, 0, 0, -10],
            [-20, -10, -10, -5, -5, -10, -10, -20]
        ],
        k: [
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [20, 20, 0, 0, 0, 0, 20, 20],
            [20, 30, 10, 0, 0, 10, 30, 20]
        ]
    };

    // État de l'IA
    let currentLevel = DIFFICULTY_LEVELS['LEVEL5'];
    let currentPersonality = PERSONALITIES['EQUILIBRE'];
    let isThinking = false;
    let nodesSearched = 0;
    let searchStartTime = 0;
    let shouldStop = false;

    /**
     * Configure le niveau de difficulté
     */
    function setLevel(levelKey) {
        if (DIFFICULTY_LEVELS[levelKey]) {
            currentLevel = DIFFICULTY_LEVELS[levelKey];
            console.log('IA Marc: Niveau configure - ' + currentLevel.name + ' (ELO ' + currentLevel.elo + ')');
            return true;
        }
        return false;
    }

    /**
     * Configure la personnalité
     */
    function setPersonality(personalityKey) {
        if (PERSONALITIES[personalityKey]) {
            currentPersonality = PERSONALITIES[personalityKey];
            console.log('IA Marc: Personnalite - ' + currentPersonality.name);
            return true;
        }
        return false;
    }

    /**
     * Vérifie si on doit arrêter la recherche
     */
    function checkStopCondition() {
        if (shouldStop) return true;
        if (nodesSearched >= currentLevel.maxNodes) return true;
        if (Date.now() - searchStartTime > currentLevel.timeLimit) return true;
        return false;
    }

    /**
     * Obtient le meilleur coup pour une position donnée
     */
    async function getBestMove(game) {
        if (isThinking) {
            console.warn('IA Marc: Deja en train de reflechir');
            return null;
        }

        isThinking = true;
        nodesSearched = 0;
        shouldStop = false;
        searchStartTime = Date.now();

        // Temps de réflexion simulé
        const minThinkTime = 200;
        const maxThinkTime = Math.min(currentLevel.timeLimit * 0.8, 2000);
        const thinkTime = Math.random() * (maxThinkTime - minThinkTime) + minThinkTime;

        return new Promise((resolve) => {
            setTimeout(() => {
                try {
                    const move = findBestMove(game);
                    const elapsed = Date.now() - searchStartTime;
                    console.log('IA Marc: Coup trouve en ' + elapsed + 'ms (' + nodesSearched + ' noeuds)');
                    isThinking = false;
                    resolve(move);
                } catch (error) {
                    console.error('IA Marc: Erreur -', error);
                    isThinking = false;
                    // Fallback: jouer un coup aléatoire
                    const moves = game.moves({ verbose: true });
                    resolve(moves.length > 0 ? moves[Math.floor(Math.random() * moves.length)] : null);
                }
            }, thinkTime);
        });
    }

    /**
     * Trouve le meilleur coup avec minimax optimisé
     */
    function findBestMove(game) {
        const moves = game.moves({ verbose: true });

        if (moves.length === 0) return null;
        if (moves.length === 1) return moves[0];

        // Chance de jouer un coup aléatoire (pour les niveaux faibles)
        if (Math.random() < currentLevel.randomMoveChance) {
            return moves[Math.floor(Math.random() * moves.length)];
        }

        let bestMove = moves[0];
        let bestScore = -Infinity;

        // Trier les coups pour améliorer l'élagage
        const sortedMoves = orderMoves(game, moves);

        // Limiter le nombre de coups analysés pour les hauts niveaux
        const maxMoves = Math.min(sortedMoves.length, 20);

        for (let i = 0; i < maxMoves; i++) {
            if (checkStopCondition()) break;

            const move = sortedMoves[i];
            game.move(move);
            const score = -minimax(game, currentLevel.depth - 1, -Infinity, Infinity, false);
            game.undo();

            // Introduire des erreurs selon le niveau
            const noise = (Math.random() - 0.5) * currentLevel.errorRate * 200;
            const adjustedScore = score + noise;

            if (adjustedScore > bestScore) {
                bestScore = adjustedScore;
                bestMove = move;
            }
        }

        return bestMove;
    }

    /**
     * Algorithme Minimax avec élagage Alpha-Beta (optimisé)
     */
    function minimax(game, depth, alpha, beta, isMaximizing) {
        nodesSearched++;

        // Vérifier les conditions d'arrêt
        if (checkStopCondition() || depth === 0) {
            return evaluatePosition(game);
        }

        if (game.game_over()) {
            if (game.in_checkmate()) {
                return isMaximizing ? -15000 + depth : 15000 - depth;
            }
            return 0;
        }

        const moves = game.moves({ verbose: true });

        // Limiter le nombre de coups analysés en profondeur
        const maxMoves = depth > 2 ? 15 : (depth > 1 ? 20 : moves.length);
        const sortedMoves = orderMoves(game, moves).slice(0, maxMoves);

        if (isMaximizing) {
            let maxScore = -Infinity;
            for (const move of sortedMoves) {
                if (checkStopCondition()) break;
                game.move(move);
                const score = minimax(game, depth - 1, alpha, beta, false);
                game.undo();
                maxScore = Math.max(maxScore, score);
                alpha = Math.max(alpha, score);
                if (beta <= alpha) break;
            }
            return maxScore;
        } else {
            let minScore = Infinity;
            for (const move of sortedMoves) {
                if (checkStopCondition()) break;
                game.move(move);
                const score = minimax(game, depth - 1, alpha, beta, true);
                game.undo();
                minScore = Math.min(minScore, score);
                beta = Math.min(beta, score);
                if (beta <= alpha) break;
            }
            return minScore;
        }
    }

    /**
     * Ordonne les coups (captures en premier)
     */
    function orderMoves(game, moves) {
        return moves.sort((a, b) => {
            let scoreA = 0, scoreB = 0;

            // Captures (MVV-LVA)
            if (a.captured) {
                scoreA += (PIECE_VALUES[a.captured] || 0) * 10 - (PIECE_VALUES[a.piece] || 0);
            }
            if (b.captured) {
                scoreB += (PIECE_VALUES[b.captured] || 0) * 10 - (PIECE_VALUES[b.piece] || 0);
            }

            // Promotions
            if (a.promotion) scoreA += 800;
            if (b.promotion) scoreB += 800;

            // Échecs (simplifié - sans vérifier réellement)
            if (a.san && a.san.includes('+')) scoreA += 50;
            if (b.san && b.san.includes('+')) scoreB += 50;

            return scoreB - scoreA;
        });
    }

    /**
     * Évalue une position (simplifié et rapide)
     */
    function evaluatePosition(game) {
        if (game.in_checkmate()) {
            return game.turn() === 'b' ? 15000 : -15000;
        }
        if (game.in_draw() || game.in_stalemate()) {
            return 0;
        }

        let score = 0;
        const board = game.board();

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board[row][col];
                if (!piece) continue;

                const pieceValue = PIECE_VALUES[piece.type] || 0;
                const pst = PST[piece.type];
                const pstRow = piece.color === 'w' ? row : 7 - row;
                const positionalValue = pst ? pst[pstRow][col] : 0;

                const totalValue = pieceValue + positionalValue;

                if (piece.color === 'b') {
                    score += totalValue;
                } else {
                    score -= totalValue;
                }
            }
        }

        return score;
    }

    /**
     * Retourne les informations du niveau actuel
     */
    function getCurrentLevel() {
        return { ...currentLevel };
    }

    /**
     * Retourne les informations de personnalité actuelle
     */
    function getCurrentPersonality() {
        return { ...currentPersonality };
    }

    /**
     * Liste tous les niveaux disponibles
     */
    function getLevels() {
        return Object.entries(DIFFICULTY_LEVELS).map(([key, level]) => ({
            key,
            ...level
        }));
    }

    /**
     * Liste toutes les personnalités disponibles
     */
    function getPersonalities() {
        return Object.entries(PERSONALITIES).map(([key, personality]) => ({
            key,
            ...personality
        }));
    }

    /**
     * Vérifie si l'IA est en train de réfléchir
     */
    function isEngineThinking() {
        return isThinking;
    }

    /**
     * Arrête la recherche
     */
    function stop() {
        shouldStop = true;
    }

    return {
        setLevel,
        setPersonality,
        getBestMove,
        getCurrentLevel,
        getCurrentPersonality,
        getLevels,
        getPersonalities,
        isThinking: isEngineThinking,
        stop
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIEngine;
}
