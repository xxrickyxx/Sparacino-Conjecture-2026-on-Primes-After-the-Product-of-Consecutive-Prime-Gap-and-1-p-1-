import numpy as np
import os

LIMIT = 10_000_000_000_000
NPZ_FILE = f"statistiche_congettura_{LIMIT}.npz"
OUT_FILE = "report_statistiche.md"

if not os.path.exists(NPZ_FILE):
    print(f"Errore: Nessun file di statistiche {NPZ_FILE} trovato.")
    exit(1)

try:
    data = np.load(NPZ_FILE, allow_pickle=True)
    hist_k = data['hist_k']
    hist_mod3 = data['hist_mod3']
    hist_d = data['hist_d'].item()
    max_k = data['max_k']
    
    total_coppie = np.sum(hist_k)
except Exception as e:
    print(f"Errore caricamento dati: {e}")
    exit(1)

# Cerchiamo l'ultimo valore non nullo per k
if np.count_nonzero(hist_k) == 0:
    last_idx = 0
else:
    last_idx = np.nonzero(hist_k)[0][-1] 

k_values = range(1, min(last_idx+1, 51)) # Limitiamo ai primi 50 k per non creare un papiro
hist_k_top = hist_k[1:51]

# Calcolo "centro di massa" di K per ogni distanza d per vedere se c'è un trend
d_x = []
k_mean_y = []
limit_elementi = list(sorted(hist_d.keys()))
for d in limit_elementi[:20]: # prime 20 distanze
    freq_arr = hist_d[d]
    totale_casi = np.sum(freq_arr)
    if totale_casi > 0:
        media = np.sum(np.arange(len(freq_arr)) * freq_arr) / totale_casi
        d_x.append(d)
        k_mean_y.append(media)

# Preparazione del testo
report = []
report.append("# Report Statistico K (Congettura di Ricky)")
report.append(f"**Coppie analizzate:** {total_coppie:,}")
report.append(f"**Limite d'analisi:** {LIMIT:,}")
report.append(f"**Massimo valore k registrato (K Max):** {last_idx}\n")

report.append("## 1. Distribuzione dei Top 20 valori di k (frequenza assoluta)")
top_20_k_indices = np.argsort(hist_k)[::-1]
# Filtra per prendere solo i K > 0 (poichè 0 è errore)
top_20_k = [idx for idx in top_20_k_indices if idx > 0][:20]

for k_id in top_20_k:
    count = hist_k[k_id]
    if count > 0:
        perc = (count / total_coppie) * 100
        report.append(f"- **k = {k_id:2}**: occorso {count:,} volte ({perc:.4f}%)")

report.append("\n## 2. Analisi K rispetto al Modulo 3 di p1")
# hist_mod3[r] dove r=1, 2
tot_mod_1 = np.sum(hist_mod3[1])
tot_mod_2 = np.sum(hist_mod3[2])
report.append(f"**Occorrenze totali quando p1 ≡ 1 (mod 3):** {tot_mod_1:,}")
report.append(f"**Occorrenze totali quando p1 ≡ 2 (mod 3):** {tot_mod_2:,}\n")

report.append("Top 5 valori di K quando **p1 ≡ 1 (mod 3)**:")
top_1 = np.argsort(hist_mod3[1])[::-1]
for i in [x for x in top_1 if x>0][:5]:
    if hist_mod3[1][i] > 0:
        report.append(f"- k = {i}: {hist_mod3[1][i]:,} volte")

report.append("\nTop 5 valori di K quando **p1 ≡ 2 (mod 3)**:")
top_2 = np.argsort(hist_mod3[2])[::-1]
for i in [x for x in top_2 if x>0][:5]:
    if hist_mod3[2][i] > 0:
        report.append(f"- k = {i}: {hist_mod3[2][i]:,} volte")

report.append("\n## 3. Relazione Media tra Distanza Primi (d) e Valore k")
report.append("| Distanza (d) | Media di K (K-Mean) |")
report.append("|--------------|---------------------|")
for d_val, k_m in zip(d_x, k_mean_y):
    report.append(f"| {d_val:<12} | {k_m:.4f}            |")

report_txt = "\n".join(report)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(report_txt)

print(f"\n{report_txt}\n")
print("-" * 50)
print(f"Salvataggio effettuato con successo in -> {OUT_FILE}")
