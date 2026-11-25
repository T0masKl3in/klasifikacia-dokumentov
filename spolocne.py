"""
Spoločné funkcie pre experimenty s klasifikáciou dokumentov (20 Newsgroups).

Každý očíslovaný skript importuje tento modul, aby všetky pokusy čítali
rovnaké dáta, rovnako ich predspracovali a rovnako merali presnosť.
"""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from scipy.stats import binom, chi2
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

KOREN = Path(__file__).resolve().parent
DATA = KOREN / "data"
VYSLEDKY = KOREN / "vysledky"
VYSLEDKY.mkdir(exist_ok=True)

NAHODA = 42  # pevné semienko, aby sa výsledky dali zopakovať
METADATA = ("headers", "footers", "quotes")


# --------------------------------------------------------------------------
# Dáta
# --------------------------------------------------------------------------
def nacitaj(cast: str, bez_metadat: bool = True):
    """Vráti (texty, triedy, názvy_tried) pre cast = 'train' alebo 'test'.

    Rozdelenie train/test je oficiálne rozdelenie podľa dátumu („bydate“):
    trénovacie sú staršie príspevky, testovacie novšie.
    bez_metadat=True odstráni hlavičky, podpisy a citované odseky.
    """
    zbierka = fetch_20newsgroups(
        subset=cast,
        remove=METADATA if bez_metadat else (),
        data_home=str(DATA),
        shuffle=True,
        random_state=NAHODA,
    )
    return list(zbierka.data), np.asarray(zbierka.target), list(zbierka.target_names)


def nacitaj_trenovacie(bez_metadat: bool = True):
    """Trénovacie dáta bez prázdnych dokumentov.

    Po odstránení metadát ostane časť príspevkov úplne prázdna (napríklad
    príspevok, ktorý bol iba citáciou). Z trénovania ich vyhodíme, lebo
    nenesú žiadnu informáciu. V teste ostávajú, lebo v praxi môžu prísť.
    """
    texty, y, nazvy = nacitaj("train", bez_metadat)
    drz = np.array([bool(TOKEN.search(t)) for t in texty])
    return [t for t, d in zip(texty, drz) if d], y[drz], nazvy, int((~drz).sum())


# --------------------------------------------------------------------------
# Predspracovanie textu
# --------------------------------------------------------------------------
TOKEN = re.compile(r"(?u)\b\w\w+\b")  # rovnaký vzor ako predvolený v scikit-learn
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
CISLO = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
STOP_SLOVA = frozenset(ENGLISH_STOP_WORDS)

_stemmer = PorterStemmer()
_kmene: dict[str, str] = {}


def kmen(slovo: str) -> str:
    """Porterov stemmer s pamäťou (to isté slovo sa nespracúva dvakrát)."""
    k = _kmene.get(slovo)
    if k is None:
        k = _stemmer.stem(slovo)
        _kmene[slovo] = k
    return k


# Varianty predspracovania: text bez úprav a potom malé písmená so všetkými
# kombináciami troch ďalších krokov (čistenie, stop slová, stemming) = 1 + 8.
# Úplné kombinácie ukážu vplyv každého kroku zvlášť aj ich súhru; pri
# postupnom pridávaní krokov by sa škodlivý krok skryl za užitočný.
def _variant(male, cistenie, stop, stemming):
    return dict(male=male, cistenie=cistenie, stop=stop, stemming=stemming)


VARIANTY = {"bez_uprav": _variant(False, False, False, False)}
for _c in (False, True):
    for _s in (False, True):
        for _k in (False, True):
            _meno = "+".join(["male"] + [n for n, z in (("cistenie", _c), ("stop", _s), ("stemming", _k)) if z])
            VARIANTY[_meno] = _variant(True, _c, _s, _k)


def predspracuj(text: str, male=True, cistenie=True, stop=True, stemming=True) -> str:
    """Prevedie dokument na reťazec tokenov oddelených medzerou.

    male      – všetko na malé písmená
    cistenie  – URL, e-maily a čísla nahradí zástupnými tokenmi
                (_url_, _email_, _cislo_) a zahodí tokeny len z podčiarkovníkov
    stop      – odstráni anglické stop slová (the, and, of, ...)
    stemming  – zredukuje slovo na kmeň (computers, computing -> comput)
    """
    if cistenie:
        text = URL.sub(" _url_ ", text)
        text = EMAIL.sub(" _email_ ", text)
        text = CISLO.sub(" _cislo_ ", text)
    if male:
        text = text.lower()
    tokeny = TOKEN.findall(text)
    if cistenie:
        tokeny = [t for t in tokeny if t.strip("_")]
    if stop:
        tokeny = [t for t in tokeny if t.lower() not in STOP_SLOVA]
    if stemming:
        tokeny = [t if t.startswith("_") else kmen(t) for t in tokeny]
    return " ".join(tokeny)


def predspracuj_vsetky(texty, **nastavenie) -> list[str]:
    return [predspracuj(t, **nastavenie) for t in texty]


# Spracované texty sa delia na tokeny už len podľa medzier.
TOKEN_PO_PREDSPRACOVANI = r"\S+"


# --------------------------------------------------------------------------
# Reprezentácie textu (z reťazca tokenov na vektor)
# --------------------------------------------------------------------------
def _tfidf(**navyse):
    return TfidfVectorizer(token_pattern=TOKEN_PO_PREDSPRACOVANI, lowercase=False, **navyse)


def _bow(**navyse):
    return CountVectorizer(token_pattern=TOKEN_PO_PREDSPRACOVANI, lowercase=False, **navyse)


REPREZENTACIE = {
    "R1_binarne_BoW": lambda: _bow(binary=True),
    "R2_pocetnosti_TF": lambda: _bow(),
    "R3_TF_normovane": lambda: _tfidf(use_idf=False),
    "R4_TF_IDF": lambda: _tfidf(),
    "R5_TF_IDF_log_tf": lambda: _tfidf(sublinear_tf=True),
    "R6_TF_IDF_log_tf_bigramy": lambda: _tfidf(sublinear_tf=True, ngram_range=(1, 2)),
    "R7_R6_bez_zriedkavych": lambda: _tfidf(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.5),
}


def zvolene_nastavenie():
    """Predspracovanie a reprezentácia vybrané v kroku 2 (vysledky/02_volba.json)."""
    volba = nacitaj_json("02_volba.json")
    return volba["predspracovanie"], VARIANTY[volba["predspracovanie"]], volba["reprezentacia"], REPREZENTACIE[volba["reprezentacia"]]


# --------------------------------------------------------------------------
# Meranie
# --------------------------------------------------------------------------
def zmeraj(y_true, y_pred) -> dict:
    """Presnosť (accuracy) a makro priemery precision, recall a F1."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
    }


def bootstrap_interval(y_true, y_pred, opakovania=1000, seed=NAHODA) -> dict:
    """95 % interval spoľahlivosti pre accuracy a makro F1 (bootstrap testovacej množiny)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1 = [], []
    for _ in range(opakovania):
        i = rng.integers(0, n, n)
        acc.append(np.mean(y_true[i] == y_pred[i]))
        f1.append(f1_score(y_true[i], y_pred[i], average="macro", zero_division=0))
    return {
        "accuracy_ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
        "f1_macro_ci95": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
    }


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemarov test: líšia sa dva modely v chybách na tých istých dokumentoch?

    Počíta len dokumenty, na ktorých sa modely nezhodli v správnosti.
    p < 0,05 znamená, že rozdiel nie je pravdepodobne náhoda.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n_ab = int(np.sum(a_ok & ~b_ok))  # A správne, B zle
    n_ba = int(np.sum(~a_ok & b_ok))  # A zle, B správne
    n = n_ab + n_ba
    p_presny = float(min(1.0, 2 * binom.cdf(min(n_ab, n_ba), n, 0.5))) if n else 1.0
    stat = (abs(n_ab - n_ba) - 1) ** 2 / n if n else 0.0
    return {
        "a_spravne_b_zle": n_ab,
        "a_zle_b_spravne": n_ba,
        "chi2_s_korekciou": float(stat),
        "p_chi2": float(chi2.sf(stat, 1)) if n else 1.0,
        "p_presny_binomicky": p_presny,
    }


# --------------------------------------------------------------------------
# Pomocné
# --------------------------------------------------------------------------
@contextmanager
def stopky(vysledok: dict, kluc: str):
    """with stopky(d, 'cas_s'): ... zapíše trvanie bloku v sekundách do d[kluc]."""
    zaciatok = time.perf_counter()
    yield
    vysledok[kluc] = round(time.perf_counter() - zaciatok, 3)


def uloz_json(meno: str, obsah) -> Path:
    cesta = VYSLEDKY / meno
    cesta.write_text(json.dumps(obsah, ensure_ascii=False, indent=2), encoding="utf-8")
    return cesta


def nacitaj_json(meno: str):
    return json.loads((VYSLEDKY / meno).read_text(encoding="utf-8"))
