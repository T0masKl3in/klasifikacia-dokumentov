# Klasifikácia dokumentov – 20 Newsgroups

Semestrálne zadanie z Manažmentu znalostí: predspracovať textové dokumenty,
vytvoriť klasifikačný model a vyhodnotiť jeho presnosť.

Autor: Tomáš Klein

Písomná časť je v súbore [`Klasifikacia_dokumentov.docx`](Klasifikacia_dokumentov.docx);
všetky čísla v nej pochádzajú z výstupov v priečinku `vysledky/`.

## Výsledok v skratke

| Model | Makro F1 (test) | Presnosť (test) |
| --- | --- | --- |
| Hybrid TF-IDF + MiniLM (začiatok), LinearSVC – vybraný krížovou validáciou | 0,720 | 73,5 % |
| Hybrid TF-IDF + MiniLM (úseky), LinearSVC | 0,723 | 73,8 % |
| Komplementový naivný Bayes (TF-IDF) | 0,708 | 72,3 % |
| Lineárne SVM (TF-IDF) | 0,696 | 71,0 % |
| MiniLM (úseky) + kNN | 0,673 | 68,5 % |
| Najčastejšia trieda (referencia) | 0,005 | 5,2 % |

Testovacia množina: 7 532 novších príspevkov bez hlavičiek, podpisov a citácií.
S ponechanými metadátami má lineárne SVM 86,9 %.

## Spustenie

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python 01_prieskum_dat.py
.venv\Scripts\python 02_predspracovanie.py
.venv\Scripts\python 03_klasifikatory.py
.venv\Scripts\python 04_embeddingy.py all-MiniLM-L6-v2
.venv\Scripts\python 05_vyhodnotenie.py
.venv\Scripts\python 06_vlakna.py
```

Skripty treba spúšťať v tomto poradí – každý číta výsledky predchádzajúceho
z priečinka `vysledky/`. Dáta 20 Newsgroups (asi 14 MB) stiahne scikit-learn
pri prvom spustení do priečinka `data/`, model all-MiniLM-L6-v2 stiahne
knižnica sentence-transformers z Hugging Face. Kódovanie embeddingov trvá na
procesore niekoľko minút (`predkoduj.py` ho vie spraviť vopred, `meraj_rychlost.py`
zmeria rýchlosť). Krok 3 sa po páde dá spustiť znova – hotové modely preskočí.

V repozitári nie sú stiahnuté dáta (`data/`), virtuálne prostredie (`.venv/`)
ani vypočítané embeddingy (`vysledky/embeddingy/`, asi 55 MB). Embeddingy
vytvorí krok 4 (alebo `predkoduj.py`); kroky 5 a 6 ich potrebujú. Ostatné
výsledky vrátane predikcií modelov na testovacej množine (`vysledky/predikcie/`)
v repozitári sú.

Najdlhšie trvá náhodný les (na notebooku s Ryzen 7 5800H 250 s na jedno
učenie, v kroku 3 šesťkrát) a kódovanie embeddingov (asi 51 dokumentov za
sekundu, teda okolo 6 minút pre začiatky dokumentov). Nespúšťajte kroky súbežne: TF-IDF s bigramami má 943 737 príznakov
a dva behy naraz sa do 16 GB nezmestia.

## Čo robí ktorý súbor

| Súbor | Obsah |
| --- | --- |
| `spolocne.py` | načítanie dát, predspracovanie textu, reprezentácie, metriky, McNemarov test, bootstrap |
| `01_prieskum_dat.py` | počty dokumentov, dĺžky, veľkosť slovníka, riedkosť |
| `02_predspracovanie.py` | 9 kombinácií predspracovania a 7 reprezentácií (5-násobná CV na trénovacích dátach) |
| `03_klasifikatory.py` | ladenie a porovnanie 6 klasifikátorov nad TF-IDF + referenčný model |
| `04_embeddingy.py` | vetné embeddingy Sentence-BERT, klasifikátory nad nimi a hybrid s TF-IDF |
| `05_vyhodnotenie.py` | súhrn, intervaly spoľahlivosti, McNemar, triedy, zámeny, slová tried, krivka učenia, metadáta |
| `06_vlakna.py` | overenie, prečo krížová validácia preceňuje presnosť (príspevky z rovnakého vlákna) |
| `predkoduj.py`, `meraj_rychlost.py` | pomocné: kódovanie embeddingov vopred, meranie rýchlosti |
| `vysledky/` | výstupy všetkých krokov (CSV, JSON, logy, predikcie) |
| `Klasifikacia_dokumentov.docx` | písomná časť zadania |

## Metodika v skratke

- Oficiálne rozdelenie „bydate“: trénovacie sú staršie príspevky, testovacie novšie.
- Hlavičky, podpisy a citácie sú odstránené (`remove=('headers','footers','quotes')`);
  311 trénovacích dokumentov, ktoré tým ostali prázdne, sa z učenia vyradí.
- Všetky voľby (predspracovanie, reprezentácia, hyperparametre) sa robia
  5-násobnou krížovou validáciou iba na trénovacej množine. Testovacia
  množina sa použije až na konečné vyhodnotenie.
- Hlavná metrika je makro F1 (každá trieda má rovnakú váhu), vedľa nej presnosť.
- Ak najlepšia hodnota hyperparametra vyjde na okraji mriežky, skript to vypíše
  (`!! ... na okraji mriežky`) a mriežka sa má rozšíriť.
- Pevné semienko náhody `NAHODA = 42` v `spolocne.py`.

## Staršie logy

`vysledky/kumulativny_navrh/` je pôvodný postupný návrh predspracovania (nahradený
všetkými kombináciami). `03_log_*` a `04_log_*` sú behy, ktoré spadli na pamäť
alebo mali optimum na okraji mriežky; ponechané pre úplnosť.
