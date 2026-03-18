import numpy as np
import matplotlib.pyplot as plt
import os

"""
Strumento di Analisi Statistiche "K" della Congettura di Ricky
Legge l'ultimo file generato dalla GPU con formato .npz e mostra i grafici interessanti
"""

LIMIT = 10_000_000_000_000 # Deve corrispondere al LIMIT dello script testcongetturadiricky.py
NPZ_FILE = f"statistiche_congettura_{LIMIT}.npz"

if not os.path.exists(NPZ_FILE):
    print(f"Errore: Nessun file di statistiche trovato! Aspetta qualche altro secondo")
    print(f"e accertati che il programma GPU stia girando regolarmente.")
    exit(1)

print(f"Lettura e analisi file: {NPZ_FILE}")

# 1. Carichiamo i dati zippati e compressi di NumPy
try:
    data = np.load(NPZ_FILE, allow_pickle=True)
    hist_k = data['hist_k']
    hist_mod3 = data['hist_mod3']
    hist_d = data['hist_d'].item()  # il dict deve usare .item() per essere estratto
    max_k = data['max_k']
    
    # Quante coppie totali sono rappresentate in questo file?
    total_coppie = np.sum(hist_k)
    print(f"File caricato con successo!")
    print(f"-> Analizzate: {total_coppie:,} coppie di numeri primi.")
    print(f"-> Valore K Massimo Storico Registrato: {max_k}")
    
except Exception as e:
    print(f"Errore durante il caricamento o decrittazione dati: {e}")
    exit(1)

# -----------------------------------------------------
# GRAFICI FIGHI DELLA DISTRIBUZIONE
# -----------------------------------------------------

plt.figure(figsize=(15, 10))

# --- GRAFICO 1: Frequenza generale di k (escludiamo i primi x valori se 0)
plt.subplot(2, 2, 1)
# Cerchiamo l'ultimo valore non nullo per zoomare il grafico al meglio invece di avere spazi vuoti giganti fino a max_k 
last_idx = np.nonzero(hist_k)[0][-1] 
plt.bar(range(1, last_idx+1), hist_k[1:last_idx+1], color='orange')
plt.title(f"Distribuzione del valore 'k' ($x - M$)")
plt.xlabel("Distanza di k dal prodotto M")
plt.ylabel("Numero di Cifre")
plt.xlim(0, last_idx + 5)


# --- GRAFICO 2: Distribuzione di k per p1 mod 3
plt.subplot(2, 2, 2)
# mod 3 può essere 0 (raro se non divisibile per 3 tranne che per p=3), 1, oppure 2.
for r in [1, 2]: 
    valori = hist_mod3[r][1:last_idx+1]
    plt.plot(range(1, last_idx+1), valori, marker='.', linestyle='none', label=f'p1 mod 3 = {r}')

plt.title("Impatto su K dal Modulo 3 di $p1$")
plt.xlabel("Valore k")
plt.ylabel("Frequenze (scala Normale)")
plt.legend()
plt.grid(alpha=0.3)


# --- GRAFICO 3: Analisi relazione tra Distanza (d) e distribuzione di k
plt.subplot(2, 2, 3)
# Prendi le distanze d più comuni per tracciarle (es. distanze d=2, d=4, d=6 tra primi gemelli ec..)
distanze_comuni = sorted(list(hist_d.keys()))[:4]  # es. prende 2, 4, 6, 8...
for d in distanze_comuni:
    if d > 0:  # Ignora zero se presente
        arr_k_per_d = hist_d[d][1:last_idx+1]
        plt.plot(range(1, last_idx+1), arr_k_per_d, marker='+', linestyle='--', alpha=0.7, label=f'd={d}')

plt.title("Influenza della Distanza fra Primi $d$ su $K$")
plt.xlabel("k ($x-M$)")
plt.ylabel("Frequenze Logaritmiche")
plt.yscale('log') # su scala log è molto più visibile questa roba
plt.legend()


# --- GRAFICO 4: K Medio all'aumentare di D
plt.subplot(2, 2, 4)
# Proviamo a calcolare un semplice "centro di massa" di K per ogni distanza d per vedere se c'è un trend
d_x = []
k_mean_y = []

# Analizziamo solo le prime N distanze sensate perché ci sono poche varianze (max sulle distanze pazzesche con pochi campioni)
limit_elementi = list(sorted(hist_d.keys()))
for d in limit_elementi[:50]:
    freq_arr = hist_d[d]
    totale_casi = np.sum(freq_arr)
    # per calcolare la media pesata di x avendo l'istogramma, moltiplichi k(indice) * frequenza
    if totale_casi > 0:
        media = np.sum(np.arange(len(freq_arr)) * freq_arr) / totale_casi
        d_x.append(d)
        k_mean_y.append(media)

plt.scatter(d_x, k_mean_y, c=d_x, cmap='viridis', edgecolors='k')
plt.title("Trend del K-Mean al variare della Distanza d")
plt.xlabel("Distanza p2-p1 (d)")
plt.ylabel("Media del Valore K ($x-M$)")
plt.grid(alpha=0.3)


plt.tight_layout()

print("--> Apre i grafici. Chiudi la finestra dell'immagine quando finisci da te per uscire.")
plt.show()
