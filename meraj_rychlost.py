"""
Pomocný skript: zmeria rýchlosť kódovania embeddingov na 1 000 testovacích
dokumentoch, keď na počítači nebeží nič iné (časy z kroku 4 môžu byť
ovplyvnené súbežnými behmi).

Použitie: python meraj_rychlost.py all-MiniLM-L6-v2 [all-mpnet-base-v2 ...]
Výstup:   vysledky/rychlost_kodovania.json
"""
import json
import sys
import time

from sentence_transformers import SentenceTransformer

from spolocne import VYSLEDKY, nacitaj

te, _, _ = nacitaj("test", bez_metadat=True)
vzorka = te[:1000]
vysledok = {}
for meno in sys.argv[1:]:
    model = SentenceTransformer(f"sentence-transformers/{meno}", device="cpu")
    model.encode(vzorka[:50], batch_size=64)  # zahriatie
    zac = time.perf_counter()
    model.encode(vzorka, batch_size=64)
    cas = time.perf_counter() - zac
    vysledok[meno] = {"dokumentov": len(vzorka), "cas_s": round(cas, 1), "dokumentov_za_s": round(len(vzorka) / cas, 1)}
    print(meno, vysledok[meno], flush=True)
(VYSLEDKY / "rychlost_kodovania.json").write_text(json.dumps(vysledok, indent=2), encoding="utf-8")
