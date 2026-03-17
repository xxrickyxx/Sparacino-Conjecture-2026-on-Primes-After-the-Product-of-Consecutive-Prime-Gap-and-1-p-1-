"""
Test della Congettura di Ricky  —  versione ottimizzata (1 MILIARDO)
====================================================================
Per ogni coppia di primi consecutivi (p1, p2) con p1 <= MAX_P1,
calcola M = d * p1 (dove d = p2 - p1) e N = ceil(ln^2(M)),
poi verifica se esiste almeno un primo nell'intervallo [M+1, M+N].

Ottimizzazioni:
  - Crivello numpy (bitwise, ~125 MB per 10^9)
  - Multiprocessing per l'analisi delle coppie
  - Progresso in tempo reale
"""

import math
import time
import sys
import numpy as np
from multiprocessing import Pool, cpu_count, Manager


# ---------------------------------------------------------------
# Crivello di Eratostene con numpy  (molto più veloce del puro Python)
# ---------------------------------------------------------------
def sieve_numpy(limit: int) -> np.ndarray:
    """Restituisce un array numpy con tutti i primi <= limit."""
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return np.nonzero(is_prime)[0]


# ---------------------------------------------------------------
# Miller-Rabin deterministico per n < 2^64
# ---------------------------------------------------------------
def miller_rabin(n: int, a: int) -> bool:
    if n % 2 == 0:
        return n == 2
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(1, s):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False


SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
MR_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    for a in MR_BASES:
        if a % n == 0:
            continue
        if not miller_rabin(n, a % n):
            return False
    return True


# ---------------------------------------------------------------
# Funzione worker per multiprocessing
# ---------------------------------------------------------------
def analyze_chunk(args):
    """Analizza un blocco di coppie di primi consecutive."""
    primes_chunk, start_idx = args
    local_successes = 0
    local_failures = []

    for k in range(len(primes_chunk) - 1):
        p1 = int(primes_chunk[k])
        p2 = int(primes_chunk[k + 1])
        d = p2 - p1
        M = d * p1

        lnM = math.log(M) if M > 0 else 0.0
        N = max(1, math.ceil(lnM * lnM))

        found = False
        for x in range(M + 1, M + N + 1):
            if is_prime(x):
                found = True
                break

        if found:
            local_successes += 1
        else:
            local_failures.append((p1, p2, d, M, N))

    return local_successes, local_failures


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    MAX_P1 = 1_000_000_000           # UN MILIARDO
    SIEVE_LIMIT = MAX_P1 + 2000      # margine per l'ultimo p2

    print("=" * 60)
    print("  TEST CONGETTURA DI RICKY  —  LIMITE: 1.000.000.000")
    print("=" * 60)
    print()

    # --- Fase 1: Crivello ---
    print(f"[1/2] Generazione primi fino a {SIEVE_LIMIT:,} (crivello numpy)...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    primes = sieve_numpy(SIEVE_LIMIT)
    t_sieve = time.perf_counter() - t0
    print(f"      Trovati {len(primes):,} primi in {t_sieve:.2f} s")
    print()

    # Filtra solo coppie con p1 <= MAX_P1
    # L'ultimo primo <= MAX_P1
    mask = primes <= MAX_P1
    last_valid = np.count_nonzero(mask)  # numero di primi <= MAX_P1
    # Servono primes[0..last_valid] (incluso il p2 dell'ultima coppia)
    primes_to_use = primes[: last_valid + 1]
    total_pairs = last_valid  # = len(primes_to_use) - 1

    print(f"[2/2] Analisi di {total_pairs:,} coppie consecutive...")
    n_workers = max(1, cpu_count() - 1)
    print(f"      Utilizzo {n_workers} processi paralleli")
    sys.stdout.flush()

    # --- Fase 2: Suddividi in chunk e analizza in parallelo ---
    CHUNK_SIZE = 50_000  # ogni worker analizza 50k coppie alla volta
    chunks = []
    for i in range(0, len(primes_to_use) - 1, CHUNK_SIZE):
        # Ogni chunk include un elemento in più per avere le coppie consecutive
        end = min(i + CHUNK_SIZE + 1, len(primes_to_use))
        chunks.append((primes_to_use[i:end], i))

    t_start = time.perf_counter()
    total_successes = 0
    all_failures = []
    pairs_done = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(analyze_chunk, chunks):
            succ, fails = result
            total_successes += succ
            all_failures.extend(fails)
            pairs_done += succ + len(fails)

            # Progresso
            pct = pairs_done / total_pairs * 100
            elapsed = time.perf_counter() - t_start
            rate = pairs_done / elapsed if elapsed > 0 else 0
            eta = (total_pairs - pairs_done) / rate if rate > 0 else 0
            print(
                f"\r      Progresso: {pairs_done:,}/{total_pairs:,} "
                f"({pct:.1f}%)  |  {rate:,.0f} coppie/s  |  "
                f"ETA: {eta:.0f}s  |  Fallimenti: {len(all_failures)}",
                end="",
            )
            sys.stdout.flush()

    t_analysis = time.perf_counter() - t_start
    print()  # nuova riga dopo il progresso

    # --- Risultati ---
    risultato_testo = []
    risultato_testo.append("=" * 60)
    risultato_testo.append("  RISULTATI (1 MILIARDO)")
    risultato_testo.append("=" * 60)
    risultato_testo.append(f"  Coppie analizzate:  {pairs_done:,}")
    risultato_testo.append(f"  Successi:           {total_successes:,}")
    risultato_testo.append(f"  Fallimenti:         {len(all_failures)}")
    risultato_testo.append(f"  Tempo crivello:     {t_sieve:.2f} s")
    risultato_testo.append(f"  Tempo analisi:      {t_analysis:.2f} s")
    risultato_testo.append(f"  Tempo totale:       {t_sieve + t_analysis:.2f} s")
    risultato_testo.append("")

    if all_failures:
        risultato_testo.append("  ⚠  FALLIMENTI TROVATI:")
        for p1, p2, d, M, N in all_failures:
            risultato_testo.append(f"     p1={p1}, p2={p2}, d={d}, M={M}, N={N}")
    else:
        risultato_testo.append("  ✅ NESSUN FALLIMENTO TROVATO!")
        risultato_testo.append("     La congettura regge su tutte le coppie testate.")
    risultato_testo.append("=" * 60)

    # Stampa a schermo
    for linea in risultato_testo:
        print(linea)

    # Scrivi su file
    with open("risultato_1miliardo.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(risultato_testo) + "\n")


if __name__ == "__main__":
    main()
