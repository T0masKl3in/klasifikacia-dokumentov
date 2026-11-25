"""
Krok 5 – vyhodnotenie a analýza.

1. Súhrnná tabuľka všetkých modelov s 95 % intervalmi spoľahlivosti (bootstrap).
2. Výber najlepšieho modelu podľa krížovej validácie (nie podľa testu).
3. McNemarove testy medzi dôležitými dvojicami modelov.
4. Matica zámen, presnosť po triedach a najčastejšie zámeny.
5. Najdôležitejšie slová každej triedy (koeficienty LinearSVC).
6. Krivka učenia: ako presnosť rastie s počtom trénovacích dokumentov.
7. Vplyv metadát: hlavičky, podpisy a citácie pri učení a pri teste.

Výstup: vysledky/05_*.csv a vysledky/05_vyhodnotenie.json
"""
import csv
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from spolocne import (
    NAHODA,
    TOKEN,
    VYSLEDKY,
    bootstrap_interval,
    kmen,
    mcnemar,
    nacitaj,
    nacitaj_json,
    nacitaj_trenovacie,
    predspracuj_vsetky,
    uloz_json,
    zmeraj,
    zvolene_nastavenie,
)

PRED = VYSLEDKY / "predikcie"


def zapis_csv(meno, hlavicka, riadky):
    with open(VYSLEDKY / meno, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hlavicka)
        w.writerows(riadky)


def subor_predikcie(kluc):
    if "|" in kluc:  # model z kroku 4: "model|sposob|klasifikator"
        m, s, k = kluc.split("|")
        k = "Hybrid" if k.startswith("Hybrid") else k
        return PRED / f"emb__{m}__{s}__{k}.npy"
    return PRED / f"{kluc}.npy"


def main():
    vystup = {}
    yte = np.load(PRED / "y_test.npy")
    _, _, nazvy = nacitaj("test", bez_metadat=True)

    # ---------------------------------------------------------------- 1. súhrn
    modely = {}
    for kluc, z in nacitaj_json("03_klasifikatory.json")["modely"].items():
        modely[kluc] = {"skupina": "TF-IDF", **z}
    for kluc, z in nacitaj_json("04_embeddingy.json")["modely"].items():
        modely[kluc] = {"skupina": "Hybrid" if "Hybrid" in kluc else "Embedding", **z}

    predikcie = {k: np.load(subor_predikcie(k)) for k in modely}
    riadky = []
    for kluc, z in modely.items():
        p = predikcie[kluc]
        m = zmeraj(yte, p)
        ci = bootstrap_interval(yte, p)
        z["test"] = m
        z["ci"] = ci
        riadky.append([
            kluc, z["skupina"], z.get("cv_f1_priemer"), round(m["accuracy"], 4),
            round(ci["accuracy_ci95"][0], 4), round(ci["accuracy_ci95"][1], 4),
            round(m["precision_macro"], 4), round(m["recall_macro"], 4), round(m["f1_macro"], 4),
            round(ci["f1_macro_ci95"][0], 4), round(ci["f1_macro_ci95"][1], 4),
            z.get("cas_ucenia_s"), z.get("cas_predikcie_s"),
        ])
    riadky.sort(key=lambda r: -(r[2] or 0))
    zapis_csv("05_suhrn.csv",
              ["model", "skupina", "cv_f1", "accuracy", "acc_ci_od", "acc_ci_do", "precision_macro",
               "recall_macro", "f1_macro", "f1_ci_od", "f1_ci_do", "cas_ucenia_s", "cas_predikcie_s"],
              riadky)
    for r in riadky:
        print(r)

    # ------------------------------------------- 2. výber podľa krížovej validácie
    def najlepsi(skupina):
        kandidati = {k: z for k, z in modely.items() if z["skupina"] == skupina}
        return max(kandidati, key=lambda k: kandidati[k]["cv_f1_priemer"])

    best_tfidf, best_emb, best_hyb = najlepsi("TF-IDF"), najlepsi("Embedding"), najlepsi("Hybrid")
    best = max(modely, key=lambda k: modely[k]["cv_f1_priemer"])
    vystup["vyber_podla_cv"] = {"TF-IDF": best_tfidf, "Embedding": best_emb, "Hybrid": best_hyb, "celkovo": best}
    print("Výber podľa CV:", vystup["vyber_podla_cv"])

    # ------------------------------------------------------------- 3. McNemar
    dvojice = [
        (best_tfidf, best_emb),
        (best_hyb, best_tfidf),
        ("LinearSVC", "MultinomialNB"),
        ("LinearSVC", "LogistickaRegresia"),
        ("ComplementNB", "MultinomialNB"),
    ]
    vystup["mcnemar"] = []
    for a, b in dvojice:
        if a in predikcie and b in predikcie and a != b:
            t = mcnemar(yte, predikcie[a], predikcie[b])
            vystup["mcnemar"].append({"A": a, "B": b, **t})
            print(f"McNemar {a} vs {b}: {t}")

    # ------------------------------------------ 4. matica zámen a triedy
    p_best = predikcie[best]
    cm = confusion_matrix(yte, p_best, labels=range(len(nazvy)))
    zapis_csv("05_matica_zamen.csv", ["skutocna", "predikovana", "pocet"],
              [[nazvy[i], nazvy[j], int(cm[i, j])] for i in range(len(nazvy)) for j in range(len(nazvy))])
    P, R, F, S = precision_recall_fscore_support(yte, p_best, labels=range(len(nazvy)), zero_division=0)
    zapis_csv("05_triedy.csv", ["trieda", "precision", "recall", "f1", "pocet_testovacich"],
              [[nazvy[i], round(P[i], 4), round(R[i], 4), round(F[i], 4), int(S[i])] for i in range(len(nazvy))])
    zameny = sorted(((int(cm[i, j]), nazvy[i], nazvy[j]) for i in range(len(nazvy))
                     for j in range(len(nazvy)) if i != j), reverse=True)[:10]
    vystup["najcastejsie_zameny"] = [{"skutocna": s, "predikovana": p, "pocet": n,
                                       "podiel_triedy": round(n / int(cm[nazvy.index(s)].sum()), 4)}
                                      for n, s, p in zameny]
    for z in vystup["najcastejsie_zameny"]:
        print("zámena", z)

    # ------------------------------------------ 5. najdôležitejšie slová tried
    meno_pred, nastavenie, _, vektorizator = zvolene_nastavenie()
    tr, ytr, _, _ = nacitaj_trenovacie(bez_metadat=True)
    te, _, _ = nacitaj("test", bez_metadat=True)
    tr_p = predspracuj_vsetky(tr, **nastavenie)
    te_p = predspracuj_vsetky(te, **nastavenie)
    param = nacitaj_json("03_klasifikatory.json")["modely"]
    svc_c = param["LinearSVC"]["najlepsie_parametre"]["C"]
    nb_alpha = param["MultinomialNB"]["najlepsie_parametre"]["alpha"]

    vek = vektorizator()
    Xtr = vek.fit_transform(tr_p)
    Xte = vek.transform(te_p)
    svc = LinearSVC(C=svc_c, max_iter=5000).fit(Xtr, ytr)
    slova = np.array(vek.get_feature_names_out())

    # kmeň -> najčastejší pôvodný tvar slova, aby sa dal výpis čítať
    tvary = defaultdict(Counter)
    if nastavenie.get("stemming"):
        for t in tr:
            for s in TOKEN.findall(t.lower()):
                tvary[kmen(s)][s] += 1

    def citatelne(term):
        return " ".join(tvary[c].most_common(1)[0][0] if tvary.get(c) else c for c in term.split())

    top = []
    for i, n in enumerate(nazvy):
        poradie = np.argsort(svc.coef_[i])[::-1][:8]
        top.append([n, ", ".join(citatelne(slova[j]) for j in poradie)])
    zapis_csv("05_top_slova.csv", ["trieda", "najsilnejsie_slova"], top)

    # ------------------------------------------------------ 6. krivka učenia
    emb_meno, emb_sposob, emb_klas = best_emb.split("|")
    Etr = np.load(VYSLEDKY / "embeddingy" / f"{emb_meno}_{emb_sposob}_train.npy")
    Ete = np.load(VYSLEDKY / "embeddingy" / f"{emb_meno}_{emb_sposob}_test.npy")
    emb_param = nacitaj_json("04_embeddingy.json")["modely"][best_emb]["najlepsie_parametre"]
    lr_c = nacitaj_json("04_embeddingy.json")["modely"][f"{emb_meno}|{emb_sposob}|LogistickaRegresia"]["najlepsie_parametre"]["C"]

    krivka = []
    rng = np.random.default_rng(NAHODA)
    for podiel in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        opakovani = 1 if podiel == 1.0 else 3
        for o in range(opakovani):
            # stratifikovaný náhodný výber podielu trénovacích dokumentov
            idx = np.concatenate([
                rng.choice(np.where(ytr == c)[0], max(2, int(round(podiel * np.sum(ytr == c)))), replace=False)
                for c in range(len(nazvy))
            ]) if podiel < 1.0 else np.arange(len(ytr))
            sub = [tr_p[i] for i in idx]
            v = vektorizator()
            Xs = v.fit_transform(sub)
            Xt = v.transform(te_p)
            vysl = {
                "MultinomialNB (TF-IDF)": MultinomialNB(alpha=nb_alpha).fit(Xs, ytr[idx]).predict(Xt),
                "LinearSVC (TF-IDF)": LinearSVC(C=svc_c, max_iter=5000).fit(Xs, ytr[idx]).predict(Xt),
                f"Logistická regresia ({emb_meno})": LogisticRegression(C=lr_c, max_iter=3000).fit(
                    Etr[idx], ytr[idx]).predict(Ete),
            }
            for model, p in vysl.items():
                krivka.append([podiel, len(idx), o, model, round(zmeraj(yte, p)["f1_macro"], 4)])
                print("krivka", krivka[-1], flush=True)
    zapis_csv("05_krivka_ucenia.csv", ["podiel", "dokumentov", "opakovanie", "model", "f1_macro_test"], krivka)

    # ------------------------------------------------------ 7. vplyv metadát
    tr_m, ytr_m, _ = nacitaj("train", bez_metadat=False)
    te_m, yte_m, _ = nacitaj("test", bez_metadat=False)
    assert np.array_equal(yte_m, yte), "poradie testovacích dokumentov sa líši"
    tr_m_p = predspracuj_vsetky(tr_m, **nastavenie)
    te_m_p = predspracuj_vsetky(te_m, **nastavenie)

    metadata = []
    for ucenie, test_text, popis in (
        (tr_m_p, te_m_p, "učenie s metadátami, test s metadátami"),
        (tr_m_p, te_p, "učenie s metadátami, test bez metadát"),
    ):
        v = vektorizator()
        Xs = v.fit_transform(ucenie)
        Xt = v.transform(test_text)
        for model, klas in (("MultinomialNB", MultinomialNB(alpha=nb_alpha)),
                            ("LinearSVC", LinearSVC(C=svc_c, max_iter=5000))):
            p = klas.fit(Xs, ytr_m).predict(Xt)
            metadata.append([popis, model, round(zmeraj(yte, p)["accuracy"], 4), round(zmeraj(yte, p)["f1_macro"], 4)])
            print("metadáta", metadata[-1], flush=True)
    for model in ("MultinomialNB", "LinearSVC"):
        z = modely[model]["test"]
        metadata.append(["učenie bez metadát, test bez metadát", model, round(z["accuracy"], 4), round(z["f1_macro"], 4)])
    zapis_csv("05_metadata.csv", ["nastavenie", "model", "accuracy", "f1_macro"], metadata)

    # pomer: o koľko je model s metadátami lepší
    vystup["prazdne_testovacie_bez_metadat"] = int(sum(1 for t in te if not TOKEN.search(t)))
    vystup["parametre"] = {"svc_C": svc_c, "nb_alpha": nb_alpha, "emb_lr_C": lr_c, "emb_najlepsi": emb_param}
    uloz_json("05_vyhodnotenie.json", vystup)


if __name__ == "__main__":
    main()
