"""
Krok 1 – prieskum dát.

Zistí, koľko je dokumentov v každej triede, aké sú dlhé, koľko ich po
odstránení metadát ostane prázdnych, aký veľký je slovník a aké riedke sú
vektory. Výstup: vysledky/01_prieskum.json a vysledky/01_triedy.csv.
"""
import csv

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from spolocne import (
    TOKEN,
    TOKEN_PO_PREDSPRACOVANI,
    VARIANTY,
    VYSLEDKY,
    nacitaj,
    predspracuj_vsetky,
    uloz_json,
)


def dlzky(texty):
    d = np.array([len(TOKEN.findall(t)) for t in texty])
    return {
        "priemer": round(float(d.mean()), 1),
        "median": int(np.median(d)),
        "p90": int(np.percentile(d, 90)),
        "max": int(d.max()),
        "prazdne": int((d == 0).sum()),
        "do_10_slov": int((d <= 10).sum()),
    }


def main():
    vystup = {}
    tr, ytr, nazvy = nacitaj("train", bez_metadat=True)
    te, yte, _ = nacitaj("test", bez_metadat=True)
    tr_m, _, _ = nacitaj("train", bez_metadat=False)
    te_m, _, _ = nacitaj("test", bez_metadat=False)

    vystup["pocet"] = {"train": len(tr), "test": len(te), "spolu": len(tr) + len(te), "triedy": len(nazvy)}
    vystup["dlzka_slov"] = {
        "train_bez_metadat": dlzky(tr),
        "test_bez_metadat": dlzky(te),
        "train_s_metadatami": dlzky(tr_m),
        "test_s_metadatami": dlzky(te_m),
    }

    # Rozdelenie tried
    c_tr = np.bincount(ytr, minlength=len(nazvy))
    c_te = np.bincount(yte, minlength=len(nazvy))
    with open(VYSLEDKY / "01_triedy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trieda", "train", "test"])
        for n, a, b in zip(nazvy, c_tr, c_te):
            w.writerow([n, int(a), int(b)])
    vystup["triedy_train_min_max"] = [int(c_tr.min()), int(c_tr.max())]
    vystup["triedy_test_min_max"] = [int(c_te.min()), int(c_te.max())]
    vystup["najmensia_trieda"] = nazvy[int(c_tr.argmin())]
    vystup["najvacsia_trieda"] = nazvy[int(c_tr.argmax())]

    # Veľkosť slovníka a riedkosť po jednotlivých krokoch predspracovania
    slovnik = {}
    for meno, nastavenie in VARIANTY.items():
        spracovane = predspracuj_vsetky(tr, **nastavenie)
        cv = CountVectorizer(token_pattern=TOKEN_PO_PREDSPRACOVANI, lowercase=False)
        X = cv.fit_transform(spracovane)
        tokenov = int(X.sum())
        slovnik[meno] = {
            "slov_v_slovniku": int(X.shape[1]),
            "tokenov_spolu": tokenov,
            "priemer_roznych_slov_na_dokument": round(float(X.getnnz(axis=1).mean()), 1),
            "nenulovych_percent": round(100 * X.nnz / (X.shape[0] * X.shape[1]), 4),
            "slov_len_v_jednom_dokumente": int((np.asarray((X > 0).sum(axis=0)).ravel() == 1).sum()),
        }
    vystup["slovnik_train"] = slovnik

    uloz_json("01_prieskum.json", vystup)

    print(f"Dokumenty: {vystup['pocet']}")
    for k, v in vystup["dlzka_slov"].items():
        print(f"  {k:22s} {v}")
    print(f"Triedy train min/max: {vystup['triedy_train_min_max']}  test: {vystup['triedy_test_min_max']}")
    for k, v in slovnik.items():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
