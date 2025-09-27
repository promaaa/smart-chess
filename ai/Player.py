import random
from Chess import Chess

class Player:
    """
    Player wrapper for Chess.
    kind: 'human', 'random', or 'auto' (auto == random by default)
    color: 'white'/'black' or True/False (True = white)
    """

    def __init__(self, color='white', kind='human'):
        self.kind = kind if kind in ('human', 'random', 'auto') else 'human'
        if isinstance(color, bool):
            self.white = color
        else:
            self.white = (color == 'white')

    # ------------------------------
    # utilitaires de conversion
    # ------------------------------
    @staticmethod
    def alg_to_sq(alg):
        """Convert 'e2' -> 0..63"""
        alg = alg.strip().lower()
        if len(alg) != 2:
            raise ValueError("Format algébrique attendu 'e2'")
        file = ord(alg[0]) - ord('a')
        rank = int(alg[1]) - 1
        if not (0 <= file <= 7 and 0 <= rank <= 7):
            raise ValueError("Case hors plateau")
        return rank * 8 + file

    @staticmethod
    def sq_to_alg(sq):
        f = sq % 8
        r = sq // 8
        return f"{chr(ord('a')+f)}{r+1}"

    # ------------------------------
    # inspection de pièce / couleur
    # ------------------------------
    def _piece_on(self, chess, square):
        mask = chess.square_mask(square)
        for p, bb in chess.bitboards.items():
            if bb & mask:
                return p
        return None

    def _piece_color_matches(self, chess, square):
        p = self._piece_on(chess, square)
        if p is None:
            return False
        return p.isupper() == self.white

    # ------------------------------
    # génération des coups LÉGAUX
    # ------------------------------
    def legal_moves(self, chess):
        """
        Return a list of legal moves as tuples (from_sq, to_sq, promotion)
        promotion is None or one-char like 'Q' or 'q'
        This enumerates pseudo-legal targets via get_all_moves then verifies legality
        by attempting the move and undoing it.
        """
        moves = []
        # iterate squares with player's pieces
        for from_sq in range(64):
            if not self._piece_color_matches(chess, from_sq):
                continue
            mask = chess.get_all_moves(from_sq)
            if mask == 0:
                continue
            # for each possible target
            for to_sq in range(64):
                if not (mask & chess.square_mask(to_sq)):
                    continue
                # detect promotions if pawn reaching last rank
                piece = self._piece_on(chess, from_sq)
                promos = [None]
                if piece and piece.lower() == 'p':
                    if (piece == 'P' and (to_sq // 8) == 7) or (piece == 'p' and (to_sq // 8) == 0):
                        promos = ['Q','R','B','N'] if piece.isupper() else ['q','r','b','n']

                for promo in promos:
                    try:
                        chess.move_piece(from_sq, to_sq, promotion=promo)
                    except ValueError:
                        # illegal (leaves roi en échec ou pseudo-illégal)
                        # ensure we didn't accidentally append to history (move_piece n'ajoute que si légal)
                        continue
                    else:
                        # accepted -> record and undo
                        moves.append((from_sq, to_sq, promo))
                        chess.undo_move()
        return moves

    # ------------------------------
    # helpers pour roque algébrique (O-O / O-O-O)
    # ------------------------------
    def _parse_castle(self, chess, token):
        """Return tuple (from_sq, to_sq, promotion) for O-O / O-O-O if legal else None"""
        token = token.strip()
        if token not in ('O-O', '0-0', 'o-o', 'O-O-O', '0-0-0', 'o-o-o'):
            return None
        white = self.white
        if white:
            king_sq = None
            if bool(chess.bitboards.get('K', 0) & chess.square_mask(4)):
                king_sq = 4
            # kingside
            if token.upper().startswith('O-O') and chess.castling_rights.get('K', False):
                return (4, 6, None)
            if token.upper().startswith('O-O-O') and chess.castling_rights.get('Q', False):
                return (4, 2, None)
        else:
            king_sq = None
            if bool(chess.bitboards.get('k', 0) & chess.square_mask(60)):
                king_sq = 60
            if token.upper().startswith('O-O') and chess.castling_rights.get('k', False):
                return (60, 62, None)
            if token.upper().startswith('O-O-O') and chess.castling_rights.get('q', False):
                return (60, 58, None)
        return None

    # ------------------------------
    # parsing d'entrée humaine
    # ------------------------------
    def _parse_human(self, s):
        """
        Accept formats:
          - 'e2e4' or 'e2 e4' or 'e2-e4'
          - promotion: 'e7e8q' or 'e7 e8 q'
          - castle: 'O-O' or 'O-O-O'
        Returns (from_sq, to_sq, promotion) or raises ValueError
        """
        s = s.strip()
        # castle
        castle = self._parse_castle_placeholder(s)
        if castle:
            return castle

        s = s.replace(' ', '').replace('-', '')
        if len(s) < 4:
            raise ValueError("Format trop court")
        frm = s[0:2]
        to = s[2:4]
        promo = s[4] if len(s) >= 5 else None
        return (Player.alg_to_sq(frm), Player.alg_to_sq(to), promo)

    def _parse_castle_placeholder(self, s):
        # normalize variants and return token if recognized
        token = s.strip()
        token_norm = token.replace('0','O').replace('o','O')
        if token_norm in ('O-O','O-O-O'):
            # convert to canonical and let _parse_castle do mapping using chess state later
            return token_norm
        return None

    # ------------------------------
    # choix du coup
    # ------------------------------
    def choose_move(self, chess):
        """
        Return a legal move tuple (from_sq, to_sq, promotion)
        according to player's kind.
        Raises ValueError if no legal moves available (mat/pat).
        """
        legal = self.legal_moves(chess)
        if not legal:
            raise ValueError("No legal moves")

        if self.kind in ('random', 'auto'):
            return random.choice(legal)

        # human: prompt loop
        while True:
            prompt = f"[{'White' if self.white else 'Black'}] Entrez votre coup (ex: e2e4, e7e8q, O-O): "
            raw = input(prompt).strip()
            if not raw:
                continue
            if raw.lower() in ('quit', 'exit'):
                raise KeyboardInterrupt()

            # castle shorthand handling
            parsed_castle = self._parse_castle_placeholder(raw)
            if parsed_castle:
                # map to a candidate using current chess state
                castle_move = self._parse_castle(chess, parsed_castle)
                if castle_move is None:
                    print("Roque non disponible dans la position actuelle.")
                    continue
                # verify castle_move present in legal moves
                for mv in legal:
                    if mv[0] == castle_move[0] and mv[1] == castle_move[1]:
                        return mv
                print("Roque interdit (case attaquée ou blocage).")
                continue

            # parse algebraic long
            try:
                frm, to, promo = self._parse_human(raw)
            except Exception as e:
                print("Parsing error:", e)
                continue

            # check against legal list
            matched = None
            for mv in legal:
                if mv[0] == frm and mv[1] == to:
                    # promotion matching: if user specified promo, match ignoring case
                    if (mv[2] is None and (promo is None)) or (mv[2] is not None and promo is not None and mv[2].upper() == promo.upper()):
                        matched = mv
                        break
                    # if move in legal with promotion options and user gave none,
                    # accept only if mv[2] is None (shouldn't happen for real promotion)
            if matched:
                return matched
            else:
                # suggest a few legal moves as examples
                examples = []
                for m in legal[:8]:
                    frm_s = Player.sq_to_alg(m[0])
                    to_s = Player.sq_to_alg(m[1])
                    promo_s = '' if m[2] is None else m[2]
                    examples.append(f"{frm_s}{to_s}{promo_s}")
                print("Coup illégal ou non reconnu. Exemples possibles :", ", ".join(examples))

    # ------------------------------
    # utilitaire de jeu: jouer un coup (pratique pour boucles externes)
    # ------------------------------
    def play_one_move(self, chess):
        """Choisit et applique un coup sur l'objet chess. Retourne le move_record tuple."""
        mv = self.choose_move(chess)
        from_sq, to_sq, promo = mv
        chess.move_piece(from_sq, to_sq, promotion=promo)
        return mv
    
import time

def pretty_move(mv):
    """Affiche un tuple (from_sq,to_sq,promo) en forme e2e4 ou e7e8Q"""
    frm, to, promo = mv
    s = f"{Player.sq_to_alg(frm)}{Player.sq_to_alg(to)}"
    if promo:
        s += promo.upper()
    return s

# -------------------------
# Exemple 1 : human vs random
# -------------------------
def play_human_vs_random():
    board = Chess()
    human = Player('white', kind='human')
    cpu = Player('black', kind='random')

    print("Début partie human (blancs) vs random (noirs). Tape 'quit' pour quitter.")
    board.print_board()
    try:
        while True:
            player = human if board.white_to_move else cpu
            try:
                mv = player.choose_move(board)
            except ValueError:
                # pas de coup légal => mat ou pat
                if board.is_in_check(board.white_to_move):
                    print(f"{'White' if board.white_to_move else 'Black'} est en échec et mat. Fin de la partie.")
                else:
                    print("Pat (draw). Fin de la partie.")
                break
            except KeyboardInterrupt:
                print("Interruption utilisateur.")
                break

            board.move_piece(mv[0], mv[1], promotion=mv[2])
            print(("Blanc" if not board.white_to_move else "Noir"), "a joué :", pretty_move(mv))
            board.print_board()
    except Exception as e:
        print("Erreur :", e)

# -------------------------
# Exemple 2 : auto vs auto (random) — démo sans interaction
# -------------------------
def play_auto_vs_auto(max_moves=200, delay=0.0):
    board = Chess()
    p1 = Player('white', kind='auto')
    p2 = Player('black', kind='auto')
    move_count = 0
    board.print_board()
    while move_count < max_moves:
        player = p1 if board.white_to_move else p2
        try:
            mv = player.choose_move(board)
        except ValueError:
            if board.is_in_check(board.white_to_move):
                print(f"Échec et mat pour {'White' if board.white_to_move else 'Black'} — fin de partie.")
            else:
                print("Pat — fin de partie.")
            break
        board.move_piece(mv[0], mv[1], promotion=mv[2])
        move_count += 1
        print(f"{move_count:03d} - { 'W' if not board.white_to_move else 'B' } played: {pretty_move(mv)}")
        if delay:
            time.sleep(delay)
        # optionally print board every N moves
        if move_count % 10 == 0:
            board.print_board()
    else:
        print("Limite de coups atteinte, fin de la partie.")
    board.print_board()

# -------------------------
# Lancer l'un ou l'autre
# -------------------------
if __name__ == "__main__":
    # Décommente la ligne que tu veux exécuter :

    # 1) partie interactive : human vs random
    play_human_vs_random()

    # 2) partie automatique : auto vs auto
    # play_auto_vs_auto(max_moves=200, delay=0.0)