"""
Krok 4 – predtrénované vetné embeddingy (Sentence-BERT) a hybrid s TF-IDF.

Každý dokument sa prevedie predtrénovanou neurónovou sieťou na hustý vektor
(384 alebo 768 čísel), ktorý zachytáva význam textu. Nad vektormi sa potom
učí obyčajný klasifikátor. Sieť sa nedoučuje (žiadny fine-tuning), takže
všetko beží na CPU.

Dva spôsoby, ako zakódovať dlhý dokument:
  * „zaciatok“ – model vidí len prvých max_seq_length tokenov, zvyšok odreže,
  * „useky“    – dokument sa rozdelí na úseky po 200 slovách, každý sa
                 zakóduje a vektory sa spriemerujú.

Hybrid: k riedkemu vektoru TF-IDF sa pripojí embedding vynásobený váhou w
a nad spojeným vektorom sa učí LinearSVC.

Použitie:  python 04_embeddingy.py [meno_modelu ...]
           (bez argumentu: all-MiniLM-L6-v2)
Výstup:    vysledky/04_embeddingy.json, vysledky/predikcie/<model>.npy,
           vysledky/embeddingy/<model>_<sposob>_<cast>.npy (vyrovnávacia pamäť)
"""
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spolocne import (
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
EMB = VYSLEDKY / "embeddingy"
EMB.mkdir(exist_ok=True)
SLOV_NA_USEK = 200


def useky(text: str) -> list[str]:
    slova = text.split()
    if not slova:
        return [""]
    return [" ".join(slova[i:i + SLOV_NA_USEK]) for i in range(0, len(slova), SLOV_NA_USEK)]


def zakoduj(model, meno_modelu, texty, sposob, cast, bez_metadat=True):
    """Vráti maticu embeddingov (riadok = dokument), L2-normovanú. Výsledok si pamätá na disku."""
    pripona = "" if bez_metadat else "_s_metadatami"
    subor = EMB / f"{meno_modelu}_{sposob}_{cast}{pripona}.npy"
    if subor.exists():
        return np.load(subor), None
    zac = time.perf_counter()
    if sposob == "zaciatok":
        E = model.encode(texty, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    else:
        vsetky, kam = [], []
        for i, t in enumerate(texty):
            u = useky(t)
            vsetky.extend(u)
            kam.extend([i] * len(u))
        V = model.encode(vsetky, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        kam = np.asarray(kam)
        E = np.zeros((len(texty), V.shape[1]), dtype=np.float32)
        np.add.at(E, kam, V)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    cas = time.perf_counter() - zac
    np.save(subor, E)
    return E, cas


def main():
    from sentence_transformers import SentenceTransformer

    mena = sys.argv[1:] or ["all-MiniLM-L6-v2"]

    # Rovnaké dokumenty ako v kroku 3: trénovacie bez prázdnych, testovacie všetky.
    tr, ytr, _, _ = nacitaj_trenovacie(bez_metadat=True)
    te, yte, _ = nacitaj("test", bez_metadat=True)

    try:
        vysledky = nacitaj_json("04_embeddingy.json")
    except FileNotFoundError:
        vysledky = {"modely": {}}

    # TF-IDF z kroku 3 pre hybrid
    _, nastavenie, _, vektorizator = zvolene_nastavenie()
    vek = vektorizator()
    tr_p = predspracuj_vsetky(tr, **nastavenie)
    Xtr_tfidf = vek.fit_transform(tr_p)
    Xte_tfidf = vek.transform(predspracuj_vsetky(te, **nastavenie))
    try:
        svc_c = nacitaj_json("03_klasifikatory.json")["modely"]["LinearSVC"]["najlepsie_parametre"]["C"]
    except (FileNotFoundError, KeyError):
        # Krok 3 ešte nedobehol: rovnaké ladenie C ako v kroku 3 (rovnaká mriežka, rovnaké foldy).
        pipe = Pipeline([("vektor", vektorizator()), ("klas", LinearSVC(max_iter=5000))])
        hl = GridSearchCV(pipe, {"klas__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
                          cv=CV, scoring="f1_macro", n_jobs=5).fit(tr_p, ytr)
        svc_c = hl.best_params_["klas__C"]
        print(f"C pre LinearSVC vyladené tu: {svc_c} (CV {hl.best_score_:.4f})", flush=True)
    vysledky["hybrid_svc_c"] = svc_c

    for meno_modelu in mena:
        model = SentenceTransformer(f"sentence-transformers/{meno_modelu}", device="cpu")
        info = vysledky.setdefault("info", {}).setdefault(meno_modelu, {})
        info["rozmer"] = int(model.get_sentence_embedding_dimension())
        info["max_tokenov"] = int(model.max_seq_length)
        print(f"== {meno_modelu} {info}", flush=True)

        for sposob in ("zaciatok", "useky"):
            Etr, cas_tr = zakoduj(model, meno_modelu, tr, sposob, "train")
            Ete, cas_te = zakoduj(model, meno_modelu, te, sposob, "test")
            if cas_tr is not None and cas_te is not None:
                info[f"cas_kodovania_{sposob}_s"] = round(cas_tr + cas_te, 1)
                info[f"dokumentov_za_s_{sposob}"] = round((len(tr) + len(te)) / (cas_tr + cas_te), 1)
            uloz_json("04_embeddingy.json", vysledky)
            print(f"   {sposob}: zakódované {Etr.shape} {Ete.shape} {info}", flush=True)

            klasifikatory = {
                "LogistickaRegresia": (LogisticRegression(max_iter=3000), {"C": [0.3, 1, 3, 10, 30, 100, 300]}),
                "LinearSVC": (LinearSVC(max_iter=5000), {"C": [0.03, 0.1, 0.3, 1, 3, 10, 30]}),
                "kNN": (KNeighborsClassifier(metric="cosine", algorithm="brute", weights="distance"),
                        {"n_neighbors": [3, 5, 10, 20, 40, 80]}),
            }
            for meno_klas, (klas, mriezka) in klasifikatory.items():
                hladanie = GridSearchCV(klas, mriezka, cv=CV, scoring="f1_macro", n_jobs=8)
                hladanie.fit(Etr, ytr)
                zac = time.perf_counter()
                pred = hladanie.predict(Ete)
                cas_pred = time.perf_counter() - zac
                kluc = f"{meno_modelu}|{sposob}|{meno_klas}"
                for k, hodnoty in mriezka.items():
                    if hladanie.best_params_[k] in (hodnoty[0], hodnoty[-1]):
                        print(f"!! {kluc}: {k} = {hladanie.best_params_[k]} je na okraji mriežky", flush=True)
                np.save(PREDIKCIE / f"emb__{meno_modelu}__{sposob}__{meno_klas}.npy", pred)
                vysledky["modely"][kluc] = {
                    "najlepsie_parametre": hladanie.best_params_,
                    "cv_f1_priemer": round(float(hladanie.best_score_), 4),
                    "cv_f1_odchylka": round(float(hladanie.cv_results_["std_test_score"][hladanie.best_index_]), 4),
                    "test": {k: round(v, 4) for k, v in zmeraj(yte, pred).items()},
                    "cas_ucenia_s": round(float(hladanie.refit_time_), 2),
                    "cas_predikcie_s": round(cas_pred, 3),
                }
                print(f"   {kluc:55s} {vysledky['modely'][kluc]['najlepsie_parametre']} "
                      f"CV {vysledky['modely'][kluc]['cv_f1_priemer']} test {vysledky['modely'][kluc]['test']}",
                      flush=True)
                uloz_json("04_embeddingy.json", vysledky)

            # Hybrid: [TF-IDF | w * embedding] -> LinearSVC s C z kroku 3, váha w z CV
            najlepsie_w, najlepsie_cv = None, -1
            VAHY = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
            for w in VAHY:
                Xtr = hstack([Xtr_tfidf, csr_matrix(w * Etr)]).tocsr()
                skore = cross_val_score(LinearSVC(C=svc_c, max_iter=5000), Xtr, ytr,
                                        cv=CV, scoring="f1_macro", n_jobs=5).mean()
                print(f"   hybrid w={w}: CV {skore:.4f}", flush=True)
                if skore > najlepsie_cv:
                    najlepsie_w, najlepsie_cv = w, skore
            if najlepsie_w in (VAHY[0], VAHY[-1]):
                print(f"!! hybrid {sposob}: w = {najlepsie_w} je na okraji mriežky", flush=True)
            Xtr = hstack([Xtr_tfidf, csr_matrix(najlepsie_w * Etr)]).tocsr()
            Xte = hstack([Xte_tfidf, csr_matrix(najlepsie_w * Ete)]).tocsr()
            zac = time.perf_counter()
            svc = LinearSVC(C=svc_c, max_iter=5000).fit(Xtr, ytr)
            cas_uc = time.perf_counter() - zac
            pred = svc.predict(Xte)
            kluc = f"{meno_modelu}|{sposob}|Hybrid_TFIDF_LinearSVC"
            np.save(PREDIKCIE / f"emb__{meno_modelu}__{sposob}__Hybrid.npy", pred)
            vysledky["modely"][kluc] = {
                "najlepsie_parametre": {"w": najlepsie_w, "C": svc_c},
                "cv_f1_priemer": round(float(najlepsie_cv), 4),
                "test": {k: round(v, 4) for k, v in zmeraj(yte, pred).items()},
                "cas_ucenia_s": round(cas_uc, 2),
            }
            print(f"   {kluc:55s} test {vysledky['modely'][kluc]['test']}", flush=True)
            uloz_json("04_embeddingy.json", vysledky)


if __name__ == "__main__":
    main()
