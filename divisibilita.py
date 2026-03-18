import pandas as pd
import numpy as np
import math

# Generiamo un mini-dataset al volo dato che non hai caricato file csv
# Simuliamo i primi N numeri primi per testare la divisibilità
print("Generando dataset dei primi piccoli per il calcolo di M...")
def prime_generator(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    return np.nonzero(is_prime)[0]

primi = prime_generator(10_000_000) # Prendi i primi fino a 10 milioni
p1 = primi[:-1]
p2 = primi[1:]
d = p2 - p1
M = d * p1

df = pd.DataFrame({'p1': p1, 'p2': p2, 'd': d, 'M': M})

print(f"Analizzando divisibilità su {len(df):,} coppie consecutive (M = p1*d)...")
print("-" * 50)

# Verifica se M+1 è divisibile per primi piccoli
primi_piccoli = [2,3,5,7,11,13,17,19,23,29,31]
for q in primi_piccoli:
    divisibili = (df['M'] + 1) % q == 0
    percentuale = divisibili.mean() * 100
    print(f"M+1 divisibile per {q:2d} : {percentuale:6.4f}%")
    if percentuale == 0:
        print(f"   ➜ EUREKA! Nessun M+1 è MAI divisibile per {q}!")