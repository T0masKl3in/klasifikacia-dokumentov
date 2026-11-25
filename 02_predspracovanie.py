"""
Krok 2 – vplyv predspracovania a reprezentácie textu.

Časť A: päť kumulatívnych variantov predspracovania (A až E) pri rovnakej
reprezentácii TF-IDF.
Časť B: sedem reprezentácií (od binárneho vreca slov po TF-IDF s bigramami)
pri najlepšom predspracovaní z časti A.

Obe časti merajú 5-násobnou krížovou validáciou IBA na trénovacej množine
(makro F1). Testovacia množina sa tu nepoužíva vôbec – slúži až na konečné
vyhodnotenie v kroku 3, aby výber nastavení nebol prispôsobený testu.
Pevné hyperparametre: MultinomialNB(alpha=0.1), LinearSVC(C=1.0).

Výstup: vysledky/02_predspracovanie.csv, 02_reprezentacia.csv, 02_volba.json
"""
import csv
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from spolocne import (
    NAHODA,
    REPREZENTACIE,
    VARIANTY,
    VYSLEDKY,
    nacitaj_trenovacie,
    predspracuj_vsetky,
    uloz_json,
)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=NAHODA)
KLASIFIKATORY = {
    "MultinomialNB": lambda: MultinomialNB(alpha=0.1),
    "LinearSVC": lambda: LinearSVC(C=1.0, max_iter=5000),
}


def ohodnot(texty, y, vektorizator):
    """Vráti riadky {klasifikator, f1_cv_priemer, f1_cv_odchylka, cas_s, velkost_slovnika}."""
    riadky = []
    velkost = len(vektorizator().fit(texty).vocabulary_)
    for meno, klas in KLASIFIKATORY.items():
        zac = time.perf_counter()
        skore = cross_val_score(
            make_pipeline(vektorizator(), klas()), texty, y, cv=CV, scoring="f1_macro", n_jobs=5
        )
        riadky.append(
            {
                "klasifikator": meno,
                "f1_cv_priemer": round(float(skore.mean()), 4),
                "f1_cv_odchylka": round(float(skore.std()), 4),
                "cas_s": round(time.perf_counter() - zac, 1),
                "velkost_slovnika": velkost,
            }
        )
    return riadky


def zapis_csv(meno, riadky):
    with open(VYSLEDKY / meno, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(riadky[0].keys()))
        w.writeheader()
        w.writerows(riadky)


def main():
    texty, y, _, vyhodenych = nacitaj_trenovacie(bez_metadat=True)
    print(f"Trénovacích dokumentov: {len(texty)} (prázdnych vyhodených: {vyhodenych})")

    # --- Časť A: predspracovanie ---------------------------------------------
    riadky_a = []
    spracovane = {}
    for meno, nastavenie in VARIANTY.items():
        spracovane[meno] = predspracuj_vsetky(texty, **nastavenie)
        for r in ohodnot(spracovane[meno], y, REPREZENTACIE["R4_TF_IDF"]):
            riadky_a.append({"variant": meno, **r})
            print(f"A {meno:16s} {r}", flush=True)
    zapis_csv("02_predspracovanie.csv", riadky_a)

    def priemer(riadky, kluc, hodnota):
        return np.mean([r["f1_cv_priemer"] for r in riadky if r[kluc] == hodnota])

    najlepsi_variant = max(VARIANTY, key=lambda v: priemer(riadky_a, "variant", v))
    print(f"Najlepšie predspracovanie (priemer oboch klasifikátorov): {najlepsi_variant}")

    # --- Časť B: reprezentácia ----------------------------------------------
    riadky_b = []
    for meno, vektorizator in REPREZENTACIE.items():
        for r in ohodnot(spracovane[najlepsi_variant], y, vektorizator):
            riadky_b.append({"reprezentacia": meno, **r})
            print(f"B {meno:26s} {r}", flush=True)
    zapis_csv("02_reprezentacia.csv", riadky_b)

    najlepsia_repr = max(REPREZENTACIE, key=lambda v: priemer(riadky_b, "reprezentacia", v))
    print(f"Najlepšia reprezentácia: {najlepsia_repr}")
    uloz_json(
        "02_volba.json",
        {
            "predspracovanie": najlepsi_variant,
            "reprezentacia": najlepsia_repr,
            "kriterium": "najvyšší priemer makro F1 z 5-násobnej CV oboch klasifikátorov",
            "prazdnych_vyhodenych_z_trenovania": vyhodenych,
        },
    )


if __name__ == "__main__":
    main()
