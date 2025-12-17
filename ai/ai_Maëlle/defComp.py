import re
import numpy as np

# recherche du fichier contenant la complexité 
with open("search_complexity_stats.txt", encoding="cp1252") as f:
    s = f.read()

#mesure
b = [float(x) for x in re.findall(r"Branching effectif moyen\s*:\s*([0-9]+(?:\.[0-9]+)?)", s)]

#affichage des valeurs obtenu
print("Nombre de positions :", len(b))
print("b_eff moyen :", np.mean(b))
print("b_eff médian :", np.median(b))
print("min / max :", min(b), "/", max(b))
