"""
Pomocný skript: vopred zakóduje dokumenty embeddingovým modelom a uloží ich
do vysledky/embeddingy/, aby krok 4 nemusel čakať. Výsledok je rovnaký, ako
keby kódoval krok 4 sám (používa jeho funkciu zakoduj).

Použitie: python predkoduj.py all-MiniLM-L6-v2 [počet_vlákien]
"""
import importlib
import json
import sys

import torch
from sentence_transformers import SentenceTransformer

from spolocne import VYSLEDKY, nacitaj, nacitaj_trenovacie

krok4 = importlib.import_module("04_embeddingy")

meno = sys.argv[1]
if len(sys.argv) > 2:
    torch.set_num_threads(int(sys.argv[2]))

model = SentenceTransformer(f"sentence-transformers/{meno}", device="cpu")
tr, _, _, _ = nacitaj_trenovacie(bez_metadat=True)
te, _, _ = nacitaj("test", bez_metadat=True)

casy = {}
for sposob in ("zaciatok", "useky"):
    _, c_tr = krok4.zakoduj(model, meno, tr, sposob, "train")
    _, c_te = krok4.zakoduj(model, meno, te, sposob, "test")
    if c_tr is not None and c_te is not None:
        casy[sposob] = {"cas_s": round(c_tr + c_te, 1),
                        "dokumentov_za_s": round((len(tr) + len(te)) / (c_tr + c_te), 1)}
    print(meno, sposob, casy.get(sposob), flush=True)

(VYSLEDKY / "embeddingy" / f"{meno}_casy.json").write_text(json.dumps(casy, indent=2), encoding="utf-8")
