# pythran export eval_core(int, int, int, int, int, int, int, int, int, int, int, int, int)


def eval_core(
    material_white, material_black,
    pst_white, pst_black,
    doubled_white, doubled_black,
    passed_white, passed_black,
    rook_open_white, rook_open_black,
    king_safety_white, king_safety_black,
    phase
):
    # Score matériel + PST
    score = (material_white - material_black) + (pst_white - pst_black)

    # Pions doublés
    score -= doubled_white * 15
    score += doubled_black * 15

    # Pions passés
    score += passed_white * 20
    score -= passed_black * 20

    # Tours sur colonnes ouvertes
    score += rook_open_white * 25
    score -= rook_open_black * 25

    # Sécurité du roi
    score += king_safety_white
    score -= king_safety_black

    # Phase (bonus fin de partie)
    score += (phase * 3)

    return score
