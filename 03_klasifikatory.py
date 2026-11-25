"""
Krok 3 – porovnanie klasifikátorov nad reprezentáciou z kroku 2.

Pre každý klasifikátor sa hyperparametre ladia mriežkovým hľadaním
(GridSearchCV) s 5-násobnou krížovou validáciou na trénovacej množine.
Najlepšie nastavenie sa natrénuje na celej trénovacej množine a až potom
sa raz vyhodnotí na testovacej množine.

Výstup: vysledky/03_klasifikatory.json, vysledky/predikcie/<model>.npy
"""
import time

import numpy as np
from joblib import Memory
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spolocne import (
    KOREN,
    NAHODA,
    VYSLEDKY,
    nacitaj,
    nacitaj_json,
    nacitaj_trenovacie,
    predspracuj_vsetky,
    uloz_json,
    zmeraj,
    zvolene_nastavenie,
)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=NAHODA)
PREDIKCIE = VYSLEDKY / "predikcie"
PREDIKCIE.mkdir(exist_ok=True)

# (klasifikátor, mriežka hyperparametrov, počet paralelných behov mriežky)
MODELY = {
    # Referencia: vždy tipne najčastejšiu triedu. Lepší model musí byť jasne nad ňou.
    "Najcastejsia_trieda": (DummyClassifier(), {"strategy": ["most_frequent"]}, 1),
    # Mriežky sú široké tak, aby najlepšia hodnota nevyšla na okraji
    # (prvý beh mal SVM na C = 3 a Bayesa na alpha = 0,003, oboje na okraji).
    "MultinomialNB": (MultinomialNB(), {"alpha": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}, 8),
    "ComplementNB": (ComplementNB(), {"alpha": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}, 8),
    "LinearSVC": (LinearSVC(max_iter=5000), {"C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}, 8),
    "kNN": (
        KNeighborsClassifier(metric="cosine", algorithm="brute", weights="distance"),
        {"n_neighbors": [3, 5, 10, 20, 40, 80]},
        8,
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=NAHODA),
        {"min_samples_leaf": [1]},
        1,
    ),
    # Logistická regresia učená stochastickým gradientovým zostupom (SGD).
    # Klasický solver L-BFGS si pamätá históriu gradientov veľkosti
    # 20 tried x ~940 tisíc príznakov (asi 3 GB na jeden beh) a na notebooku
    # so 16 GB padal aj pri 2 paralelných behoch (WinError 1450). SGD minimalizuje
    # tú istú logistickú stratu s L2 regularizáciou (alpha zodpovedá 1 / (C * n)),
    # triedy rieši ako jedna proti ostatným.
    "LogistickaRegresia": (
        SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4, random_state=NAHODA),
        {"alpha": [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]},
        8,
    ),
}


def main():
    meno_pred, nastavenie, meno_repr, vektorizator = zvolene_nastavenie()
    print(f"Predspracovanie {meno_pred}, reprezentácia {meno_repr}")

    tr, ytr, nazvy, _ = nacitaj_trenovacie(bez_metadat=True)
    te, yte, _ = nacitaj("test", bez_metadat=True)
    tr = predspracuj_vsetky(tr, **nastavenie)
    te = predspracuj_vsetky(te, **nastavenie)
    np.save(PREDIKCIE / "y_test.npy", yte)

    pamat = Memory(KOREN / ".cache", verbose=0)  # vektorizácia sa pre rovnaký fold nepočíta znova
    vysledky = {"predspracovanie": meno_pred, "reprezentacia": meno_repr, "modely": {}}
    # Pokračovanie po páde: model s rovnakým nastavením, ktorý už má výsledok
    # aj uložené predikcie, sa neučí znova.
    try:
        stare = nacitaj_json("03_klasifikatory.json")
        if (stare["predspracovanie"], stare["reprezentacia"]) == (meno_pred, meno_repr):
            vysledky["modely"] = {m: z for m, z in stare["modely"].items()
                                  if m in MODELY and (PREDIKCIE / f"{m}.npy").exists()}
    except FileNotFoundError:
        pass

    for meno, (klasifikator, mriezka, paralelne) in MODELY.items():
        if meno in vysledky["modely"]:
            print(f"{meno:20s} už hotový, preskakujem", flush=True)
            continue
        pipe = Pipeline([("vektor", vektorizator()), ("klas", klasifikator)], memory=pamat)
        mriezka = {f"klas__{k}": v for k, v in mriezka.items()}
        hladanie = GridSearchCV(pipe, mriezka, cv=CV, scoring="f1_macro", n_jobs=paralelne, refit=True)
        zac = time.perf_counter()
        hladanie.fit(tr, ytr)
        cas_hladania = time.perf_counter() - zac

        zac = time.perf_counter()
        pred = hladanie.predict(te)
        cas_predikcie = time.perf_counter() - zac
        np.save(PREDIKCIE / f"{meno}.npy", pred)

        i = hladanie.best_index_
        zaznam = {
            "najlepsie_parametre": {k.replace("klas__", ""): v for k, v in hladanie.best_params_.items()},
            "cv_f1_priemer": round(float(hladanie.cv_results_["mean_test_score"][i]), 4),
            "cv_f1_odchylka": round(float(hladanie.cv_results_["std_test_score"][i]), 4),
            "mriezka": [
                {"parametre": {k.replace("klas__", ""): v for k, v in p.items()},
                 "cv_f1": round(float(s), 4)}
                for p, s in zip(hladanie.cv_results_["params"], hladanie.cv_results_["mean_test_score"])
            ],
            "test": {k: round(v, 4) for k, v in zmeraj(yte, pred).items()},
            "cas_ucenia_s": round(float(hladanie.refit_time_), 2),
            "cas_predikcie_s": round(cas_predikcie, 2),
            "cas_celeho_hladania_s": round(cas_hladania, 1),
        }
        # Upozornenie, ak najlepšia hodnota leží na okraji mriežky (optimum môže byť mimo nej)
        for k, hodnoty in mriezka.items():
            if len(hodnoty) > 2 and hladanie.best_params_[k] in (hodnoty[0], hodnoty[-1]):
                zaznam["na_okraji_mriezky"] = k.replace("klas__", "")
                print(f"!! {meno}: {k} = {hladanie.best_params_[k]} je na okraji mriežky", flush=True)
        vysledky["modely"][meno] = zaznam
        print(f"{meno:20s} {zaznam['najlepsie_parametre']} CV F1 {zaznam['cv_f1_priemer']} "
              f"test {zaznam['test']} učenie {zaznam['cas_ucenia_s']} s", flush=True)
        uloz_json("03_klasifikatory.json", vysledky)  # priebežne, keby beh spadol


if __name__ == "__main__":
    main()
