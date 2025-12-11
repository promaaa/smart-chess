/**
 * Smart Chess - AI Engine Module
 * Connecteur vers l'IA Marc V2 via le serveur Python
 */

const AIEngine = (function () {
    'use strict';

    const API_BASE = '';  // Même origine

    // Cache des niveaux et personnalités
    let cachedLevels = null;
    let cachedPersonalities = null;
    let currentLevel = 'LEVEL5';
    let currentPersonality = 'EQUILIBRE';
    let isThinking = false;
    let useBackend = true;  // Utiliser le backend Python

    // Niveaux par défaut (fallback si le serveur ne répond pas)
    const DEFAULT_LEVELS = {
        'LEVEL1': { name: 'Niveau 1 (Débutant)', elo: 400 },
        'LEVEL2': { name: 'Niveau 2 (Novice)', elo: 600 },
        'LEVEL3': { name: 'Niveau 3 (Occasionnel)', elo: 800 },
        'LEVEL4': { name: 'Niveau 4 (Amateur)', elo: 1000 },
        'LEVEL5': { name: 'Niveau 5 (Club débutant)', elo: 1200 },
        'LEVEL6': { name: 'Niveau 6 (Club intermédiaire)', elo: 1400 },
        'LEVEL7': { name: 'Niveau 7 (Club avancé)', elo: 1600 },
        'LEVEL8': { name: 'Niveau 8 (Maximum)', elo: 1700 }
    };

    const DEFAULT_PERSONALITIES = {
        'EQUILIBRE': { name: 'Équilibré', description: 'Jeu équilibré' },
        'AGRESSIF': { name: 'Agressif', description: 'Attaque constamment' },
        'DEFENSIF': { name: 'Défensif', description: 'Jeu solide' },
        'POSITIONNEL': { name: 'Positionnel', description: 'Contrôle de l\'espace' },
        'TACTIQUE': { name: 'Tactique', description: 'Cherche les combinaisons' },
        'MATERIALISTE': { name: 'Matérialiste', description: 'Privilégie le matériel' }
    };

    /**
     * Vérifie si le backend est disponible
     */
    async function checkBackend() {
        try {
            var response = await fetch(API_BASE + '/api/status', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            if (response.ok) {
                var data = await response.json();
                useBackend = data.status === 'ok' && data.engine;
                console.log('IA Marc Backend: ' + (useBackend ? 'Connecté' : 'Non disponible'));
                return useBackend;
            }
        } catch (e) {
            console.log('IA Marc Backend: Non disponible (mode local)');
            useBackend = false;
        }
        return false;
    }

    /**
     * Configure le niveau
     */
    async function setLevel(levelKey) {
        currentLevel = levelKey;

        if (useBackend) {
            try {
                var response = await fetch(API_BASE + '/api/set_level', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ level: levelKey })
                });
                var data = await response.json();
                if (data.success) {
                    console.log('IA Marc: Niveau configuré - ' + levelKey);
                    return true;
                }
            } catch (e) {
                console.error('Erreur set_level:', e);
            }
        }
        return true;
    }

    /**
     * Configure la personnalité
     */
    function setPersonality(personalityKey) {
        currentPersonality = personalityKey;
        console.log('IA Marc: Personnalité - ' + personalityKey);
        return true;
    }

    /**
     * Obtient le meilleur coup depuis le backend IA Marc V2
     */
    async function getBestMove(game) {
        if (isThinking) {
            return null;
        }

        isThinking = true;

        try {
            // Essayer le backend Python
            if (useBackend) {
                var response = await fetch(API_BASE + '/api/move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fen: game.fen() })
                });

                var data = await response.json();

                if (data.success && data.move) {
                    console.log('IA Marc V2: ' + data.move.san + ' (depuis le serveur)');

                    if (data.stats) {
                        console.log('Stats:', data.stats);
                    }

                    isThinking = false;
                    return {
                        from: data.move.from,
                        to: data.move.to,
                        promotion: data.move.promotion,
                        san: data.move.san
                    };
                }
            }

            // Fallback: coup simple (première capture ou premier coup)
            console.log('IA Marc: Mode fallback (serveur non disponible)');
            var moves = game.moves({ verbose: true });

            // Préférer les captures
            var captures = moves.filter(function (m) { return m.captured; });
            var move = captures.length > 0 ? captures[0] : moves[0];

            isThinking = false;
            return move;

        } catch (error) {
            console.error('Erreur getBestMove:', error);
            isThinking = false;

            // Fallback
            var moves = game.moves({ verbose: true });
            return moves.length > 0 ? moves[0] : null;
        }
    }

    /**
     * Retourne les infos du niveau actuel
     */
    function getCurrentLevel() {
        var level = DEFAULT_LEVELS[currentLevel] || DEFAULT_LEVELS['LEVEL5'];
        return {
            key: currentLevel,
            name: level.name,
            elo: level.elo
        };
    }

    /**
     * Retourne les infos de la personnalité actuelle
     */
    function getCurrentPersonality() {
        var p = DEFAULT_PERSONALITIES[currentPersonality] || DEFAULT_PERSONALITIES['EQUILIBRE'];
        return {
            key: currentPersonality,
            name: p.name,
            description: p.description
        };
    }

    /**
     * Liste les niveaux
     */
    async function getLevels() {
        // Essayer de récupérer depuis le serveur
        if (useBackend && !cachedLevels) {
            try {
                var response = await fetch(API_BASE + '/api/levels');
                var data = await response.json();
                if (data.levels) {
                    cachedLevels = data.levels;
                    return cachedLevels;
                }
            } catch (e) {
                console.log('Utilisation des niveaux par défaut');
            }
        }

        if (cachedLevels) {
            return cachedLevels;
        }

        // Niveaux par défaut
        return Object.keys(DEFAULT_LEVELS).map(function (key) {
            return {
                key: key,
                name: DEFAULT_LEVELS[key].name,
                elo: DEFAULT_LEVELS[key].elo
            };
        });
    }

    /**
     * Liste les personnalités
     */
    function getPersonalities() {
        return Object.keys(DEFAULT_PERSONALITIES).map(function (key) {
            return {
                key: key,
                name: DEFAULT_PERSONALITIES[key].name,
                description: DEFAULT_PERSONALITIES[key].description
            };
        });
    }

    /**
     * Vérifie si l'IA réfléchit
     */
    function isEngineThinking() {
        return isThinking;
    }

    /**
     * Initialise la connexion au backend
     */
    async function init() {
        await checkBackend();
        return useBackend;
    }

    // Vérifier le backend au chargement
    if (typeof window !== 'undefined') {
        window.addEventListener('load', function () {
            checkBackend();
        });
    }

    return {
        init: init,
        setLevel: setLevel,
        setPersonality: setPersonality,
        getBestMove: getBestMove,
        getCurrentLevel: getCurrentLevel,
        getCurrentPersonality: getCurrentPersonality,
        getLevels: getLevels,
        getPersonalities: getPersonalities,
        isThinking: isEngineThinking,
        checkBackend: checkBackend
    };
})();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIEngine;
}
