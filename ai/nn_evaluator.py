import numpy as np
from Chess import Chess

class NeuralNetworkEvaluator:
    """
    Évaluateur de position d'échecs basé sur un réseau de neurones.
    L'architecture est séparée des poids pour permettre le chargement/sauvegarde.
    """
    def __init__(self, weights1, biases1, weights2, biases2, weights3, biases3):
        """
        Initialise le réseau avec deux hidden layers.
        """
        self.weights1 = weights1
        self.biases1 = biases1
        self.weights2 = weights2
        self.biases2 = biases2
        self.weights3 = weights3
        self.biases3 = biases3

        # Vérifications de dimensions
        assert self.weights1.shape[1] == self.biases1.shape[1], "Dimension mismatch in hidden layer 1"
        assert self.weights2.shape[0] == self.weights1.shape[1], "Dimension mismatch between layer 1 and 2"
        assert self.weights2.shape[1] == self.biases2.shape[1], "Dimension mismatch in hidden layer 2"
        assert self.weights3.shape[0] == self.weights2.shape[1], "Dimension mismatch between layer 2 and output"

        # Mapping des pièces (inchangé)
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        self.input_size = self.weights1.shape[0]

    @staticmethod
    def create_untrained_network(input_size=768, hidden_size=256, output_size=1):
        """
        Crée un réseau à deux hidden layers de 128 neurones chacun avec initialisation He.
        """
        w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        b1 = np.zeros((1, hidden_size))
        w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        b2 = np.zeros((1, hidden_size))
        w3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        b3 = np.zeros((1, output_size))
        return NeuralNetworkEvaluator(w1, b1, w2, b2, w3, b3)

    def _encode_board(self, chess_instance: Chess) -> np.ndarray:
        # Cette fonction reste identique
        board_vector = np.zeros(self.input_size)
        for piece_char, bitboard in chess_instance.bitboards.items():
            if bitboard == 0: continue
            piece_index = self.piece_to_index[piece_char]
            temp_bb = int(bitboard)
            while temp_bb > 0:
                square = (temp_bb & -temp_bb).bit_length() - 1
                vector_position = piece_index * 64 + square
                board_vector[vector_position] = 1.0
                temp_bb &= temp_bb - 1
        return board_vector

    def evaluate_position(self, chess_instance: Chess) -> float:
        input_vector = self._encode_board(chess_instance)
        h1 = np.maximum(0, np.dot(input_vector, self.weights1) + self.biases1)
        h2 = np.maximum(0, np.dot(h1, self.weights2) + self.biases2)
        output_layer_input = np.dot(h2, self.weights3) + self.biases3
        normalized_score = output_layer_input[0][0]
        EVAL_SCALE_FACTOR = 1000.0
        centipawn_score = normalized_score * EVAL_SCALE_FACTOR
        return centipawn_score

def save_weights(evaluator: NeuralNetworkEvaluator, filename: str):
    """Sauvegarde les poids et biais de l'évaluateur dans un fichier .npz."""
    np.savez(filename,
             w1=evaluator.weights1,
             b1=evaluator.biases1,
             w2=evaluator.weights2,
             b2=evaluator.biases2,
             w3=evaluator.weights3,
             b3=evaluator.biases3)
    print(f"Poids sauvegardés dans {filename}")

def load_evaluator_from_file(filename: str) -> NeuralNetworkEvaluator:
    """Charge les poids d'un fichier et crée une instance de l'évaluateur."""
    data = np.load(filename)
    return NeuralNetworkEvaluator(
        weights1=data['w1'],
        biases1=data['b1'],
        weights2=data['w2'],
        biases2=data['b2'],
        weights3=data['w3'],
        biases3=data['b3']
    )

# --- COMMENT UTILISER CETTE NOUVELLE STRUCTURE ---
if __name__ == '__main__':
    WEIGHTS_FILE = "chess_nn_weights.npz"
    game = Chess()

    # --- SCÉNARIO 1 : Démarrer un entraînement (créer un réseau vierge) ---
    print("--- Création d'un nouveau réseau non entraîné ---")
    untrained_evaluator = NeuralNetworkEvaluator.create_untrained_network()
    
    # Évaluer la position de départ (score aléatoire)
    score1 = untrained_evaluator.evaluate_position(game)
    print(f"Score du réseau vierge : {score1:.2f}")

    # Sauvegarder ses poids initiaux pour une utilisation future
    save_weights(untrained_evaluator, WEIGHTS_FILE)

    # --- SCÉNARIO 2 : Utiliser le moteur (charger un réseau entraîné) ---
    print(f"\n--- Chargement du réseau depuis le fichier '{WEIGHTS_FILE}' ---")
    
    # Supposons que notre programme vient de démarrer. On charge les poids.
    production_evaluator = load_evaluator_from_file(WEIGHTS_FILE)
    
    # L'évaluation doit être IDENTIQUE à celle du réseau que nous avons sauvegardé
    score2 = production_evaluator.evaluate_position(game)
    print(f"Score du réseau chargé : {score2:.2f}")
    assert np.isclose(score1, score2) # Vérifie que les scores sont les mêmes
    
    print("\nLe chargement des poids a fonctionné correctement !")