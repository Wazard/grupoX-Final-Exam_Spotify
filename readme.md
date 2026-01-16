# AI DJ 🎧  
### Online Music Recommendation System

AI DJ è un sistema di **raccomandazione musicale online** che simula il comportamento di un *DJ intelligente*, capace di adattarsi dinamicamente ai gusti dell’utente attraverso interazioni successive.

Il sistema apprende in tempo reale tramite **feedback binario** (like / dislike) e utilizza un algoritmo di **Online Machine Learning** basato su **Linear Thompson Sampling**, un approccio che combina **Multi-Armed Bandit** e **inferenza bayesiana** per bilanciare **esplorazione** e **sfruttamento**.

---

## Dataset

Il progetto utilizza il classico **dataset Spotify Audio Features**, in cui ogni canzone è descritta tramite:

- **Audio features** come `danceability`, `energy`, `loudness`, `speechiness`,  
  `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- **Metadati** quali `popularity`, genere musicale e artista

Queste feature rappresentano il contesto su cui opera l’algoritmo di raccomandazione.

---

## Preprocessing e Feature Engineering

Il dataset viene preprocessato per migliorare la qualità e l’espressività delle informazioni:

- Rimozione di outlier estremi e gestione dei duplicati
- Encoding dei generi musicali
- **Feature engineering**, includendo:
  - feature di interazione
  - binning di variabili continue
  - feature binarie
  - statistiche derivate (es. popolarità media per artista)

Per ridurre la dimensionalità e il rumore:
- Applicazione di **PCA**, mantenendo circa il **90% della varianza**

---

## Clustering e Cold Start

Per rendere l’esplorazione più efficiente e gestire il **cold start**:

- Clustering delle canzoni tramite **K-Means** (k ≈ 10)
- I cluster rappresentano gruppi musicali semanticamente coerenti
- Le raccomandazioni iniziali sono bilanciate tra cluster e basate sulla popolarità

Il clustering riduce lo spazio di esplorazione e migliora la qualità delle prime raccomandazioni.

---

## Modello di Raccomandazione

Il modello base è un **Thompson Sampling** con reward Bernoulliano:

- Prior **Beta**
- Aggiornamento **online** del posterior dopo ogni interazione
- Esplorazione guidata dall’incertezza

Sono state esplorate anche varianti più strutturate:

- **Hierarchical Thompson Sampling**, con selezione a livello di cluster e intra-cluster
- **Cluster Thompson Sampling Hybrid**, che combina:
  - selezione sui cluster
  - ranking adattivo delle canzoni all’interno del cluster

---

## Modellazione dell’Utente e Valutazione

Le prestazioni del sistema sono valutate tramite **utenti simulati**:

- Modelli deterministici (basati sul genere)
- Modelli probabilistici più realistici

I test mostrano che il sistema:
- converge rapidamente a raccomandazioni corrette
- si adatta a **cambi di gusto** sia netti che graduali
- si comporta come atteso in presenza di utenti casuali

Il confronto con baseline **random-based**, **popularity-based** e **cluster-based** evidenzia i vantaggi dell’approccio online.

---

## Conclusioni

AI DJ dimostra come un sistema di raccomandazione musicale **online**, basato su Thompson Sampling e arricchito da clustering e feature engineering, possa adattarsi efficacemente a utenti dinamici in scenari realistici.

Il progetto è facilmente estendibile a contesti multi-utente e a test con utenti reali.
