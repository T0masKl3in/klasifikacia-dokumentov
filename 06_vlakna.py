"""
Krok 6 – prečo krížová validácia preceňuje presnosť (overenie hypotézy).

Hypotéza: pri náhodnom delení trénovacej množiny na foldy má dokument
medzi najbližšími susedmi často príspevok z toho istého diskusného vlákna
(rovnaký predmet „Subject“). Testovacie dokumenty sú novšie, ich vlákna
v trénovacej množine väčšinou nie sú, takže na teste je model horší.

Meranie: pre každý dokument nájdeme najbližší trénovací dokument
(kosínusová podobnosť embeddingov MiniLM, ten istý dokument sa nepočíta)
a zistíme, či má rovnaký predmet a rovnakú triedu.
  * trénovací -> iný trénovací  (situácia v krížovej validácii)
  * testovací -> trénovací      (situácia pri teste)
Predmet sa berie z hlavičky Subject bez „Re:“; v samotnom texte modelov nie je.

Výstup: vysledky/06_vlakna.json
"""
import re

import numpy as np

from spolocne import TOKEN, VYSLEDKY, nacitaj, uloz_json

RE = re.compile(r"^\s*((re|fwd?)\s*:\s*)+", re.IGNORECASE)


def predmet(text_s_hlavickou: str) -> str:
    for riadok in text_s_hlavickou.splitlines():
        if riadok.lower().startswith("subject:"):
            return RE.sub("", riadok[8:].strip()).lower()
        if not riadok.strip():  # koniec hlavičky
            break
    return ""


def main():
    # rovnaké poradie dokumentov ako v ostatných krokoch (rovnaké semienko)
    tr_bez, ytr, _ = nacitaj("train", bez_metadat=True)
    te_bez, yte, _ = nacitaj("test", bez_metadat=True)
    tr_s, ytr_s, _ = nacitaj("train", bez_metadat=False)
    te_s, yte_s, _ = nacitaj("test", bez_metadat=False)
    assert np.array_equal(ytr, ytr_s) and np.array_equal(yte, yte_s)

    drz = np.array([bool(TOKEN.search(t)) for t in tr_bez])  # ako v kroku 2-4
    p_tr = np.array([predmet(t) for t in tr_s])[drz]
    p_te = np.array([predmet(t) for t in te_s])
    ytr = ytr[drz]

    Etr = np.load(VYSLEDKY / "embeddingy" / "all-MiniLM-L6-v2_zaciatok_train.npy")
    Ete = np.load(VYSLEDKY / "embeddingy" / "all-MiniLM-L6-v2_zaciatok_test.npy")
    assert len(Etr) == len(ytr) and len(Ete) == len(yte)

    def najblizsi(E, vylucit_seba):
        idx = np.empty(len(E), dtype=int)
        for zac in range(0, len(E), 1000):
            S = E[zac:zac + 1000] @ Etr.T
            if vylucit_seba:
                S[np.arange(S.shape[0]), np.arange(zac, zac + S.shape[0])] = -np.inf
            idx[zac:zac + 1000] = S.argmax(axis=1)
        return idx

    vystup = {}
    for meno, E, P, Y, seba in (("trenovaci_k_trenovaciemu", Etr, p_tr, ytr, True),
                                ("testovaci_k_trenovaciemu", Ete, p_te, yte, False)):
        n = najblizsi(E, seba)
        maPredmet = P != ""
        rovnaky = (P == p_tr[n]) & maPredmet
        spravne = Y == ytr[n]
        vystup[meno] = {
            "dokumentov": int(len(E)),
            "rovnaky_predmet_podiel": round(float(rovnaky.mean()), 4),
            "sused_ma_rovnaku_triedu_podiel": round(float(spravne.mean()), 4),
            "rovnaka_trieda_ak_rovnaky_predmet": round(float(spravne[rovnaky].mean()), 4),
            "rovnaka_trieda_ak_iny_predmet": round(float(spravne[~rovnaky].mean()), 4),
        }
        print(meno, vystup[meno], flush=True)

    # Kontrola: náhodný trénovací dokument namiesto najbližšieho (má byť takmer 0)
    rng = np.random.default_rng(0)
    nahodny = rng.integers(0, len(p_tr), len(p_tr))
    vystup["kontrola_nahodny_sused_rovnaky_predmet"] = round(
        float(((p_tr == p_tr[nahodny]) & (p_tr != "") & (nahodny != np.arange(len(p_tr)))).mean()), 4)
    vystup["bez_predmetu"] = {"train": int((p_tr == "").sum()), "test": int((p_te == "").sum())}
    print("kontrola (náhodný sused):", vystup["kontrola_nahodny_sused_rovnaky_predmet"],
          "bez predmetu:", vystup["bez_predmetu"])

    # Koľko testovacích predmetov (vlákien) sa vyskytuje aj v trénovacej množine
    vystup["testovacich_s_predmetom_aj_v_treningu"] = round(float(np.isin(p_te, p_tr[p_tr != ""]).mean()), 4)
    print("podiel testovacích dokumentov, ktorých predmet je aj v tréningu:",
          vystup["testovacich_s_predmetom_aj_v_treningu"])
    uloz_json("06_vlakna.json", vystup)


if __name__ == "__main__":
    main()
