import math
import sys
import time
import numpy as np
import datetime
import cupy as cp
import os

"""
Test della Congettura di Ricky  —  Versione CUDA (GPU NVIDIA) + Miller-Rabin! + Statistiche
===================================================================================
Altissime Prestazioni:
 - Sfrutta la tua RTX 5060 Ti in linguaggio CUDA C++ nativo! Millioni di thread simultanei.
 - Matematica Miller-Rabin su GPU per accelerare 1000x!
 - Estrazione e raccolta dati avanzata: calcolo del valore "k", accumulo su istogrammi 
   per studiare le correlazioni con "d" e rotazioni modulo 3 di p1.
"""

cuda_kernel_code = r'''
__device__ unsigned long long mul_mod(unsigned long long a, unsigned long long b, unsigned long long m) {
    unsigned long long res = 0;
    a %= m;
    while (b > 0) {
        if (b & 1) {
            res += a;
            if (res >= m) res -= m;
        }
        b >>= 1;
        a += a;
        if (a >= m) a -= m;
    }
    return res;
}

__device__ unsigned long long pow_mod(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) res = mul_mod(res, base, mod);
        base = mul_mod(base, base, mod);
        exp >>= 1;
    }
    return res;
}

__device__ bool miller_rabin(unsigned long long n, unsigned long long a) {
    unsigned long long d = n - 1;
    int s = 0;
    while ((d % 2) == 0) {
        d /= 2;
        s++;
    }
    unsigned long long x = pow_mod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int r = 1; r < s; r++) {
        x = mul_mod(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

extern "C" __global__
void analyze_pairs(const unsigned long long* p1_arr, const unsigned long long* p2_arr, unsigned short* k_arr, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        unsigned long long p1 = p1_arr[idx];
        unsigned long long p2 = p2_arr[idx];
        
        unsigned long long d = p2 - p1;
        unsigned long long M = d * p1;
        
        // N = ceil(ln(M)^2)
        double lnM = log((double)M);
        if (lnM < 0.0) lnM = 0.0;
        unsigned long long N = (unsigned long long)ceil(lnM * lnM);
        if (N < 1) N = 1;
        
        unsigned long long start_check = M + 1;
        unsigned long long end_check = M + N;
        
        unsigned short k_val = 0;
        
        if (start_check <= 2 && end_check >= 2) {
            k_val = (unsigned short)(2 - M);
        } else {
            if (start_check % 2 == 0) start_check++;
            unsigned long long bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
            
            for (unsigned long long x = start_check; x <= end_check; x += 2) {
                bool is_p = true;
                if (x < 2) is_p = false;
                else if (x == 2 || x == 3 || x == 5 || x == 7) is_p = true;
                else if (x % 2 == 0 || x % 3 == 0 || x % 5 == 0 || x % 7 == 0) is_p = false;
                else {
                    for (int i = 0; i < 7; i++) {
                        unsigned long long a = bases[i];
                        if (x <= a) break;
                        if (a % x == 0) continue;
                        if (!miller_rabin(x, a % x)) {
                            is_p = false;
                            break;
                        }
                    }
                }
                
                if (is_p) {
                    k_val = (unsigned short)(x - M);
                    break;
                }
            }
        }
        
        k_arr[idx] = k_val;
    }
}
'''

analyze_kernel = cp.RawKernel(cuda_kernel_code, 'analyze_pairs')

# Sieve Generation part remains the same as previously optimized
def prime_generator(limit):
    sqrt_limit = math.isqrt(limit) + 1
    
    is_base = np.ones(sqrt_limit + 1, dtype=np.bool_)
    is_base[0] = is_base[1] = False
    for i in range(2, math.isqrt(sqrt_limit) + 1):
        if is_base[i]:
            is_base[i*i::i] = False
    base_primes = np.nonzero(is_base)[0]
    base_primes = base_primes[1:]
    
    # Chunk Size da 25M
    S = 25_000_000 
    sieve = np.empty(S, dtype=np.bool_)
    
    last_prime = 2
    
    for low in range(0, limit, 2 * S):
        max_val = min(low + 2 * S - 1, limit)
        if max_val <= low: break
        
        max_k = (max_val - low - 1) // 2
        sieve[:max_k+1] = True
        
        if low == 0:
            sieve[0] = False
            
        for p in base_primes:
            start_val = (low + p - 1) // p * p
            if start_val % 2 == 0: start_val += p
            if start_val == p: start_val += 2 * p
            
            if start_val <= max_val:
                k_start = (start_val - low - 1) // 2
                sieve[k_start:max_k+1:p] = False
                
        primes_in_chunk = low + 2 * np.nonzero(sieve[:max_k+1])[0] + 1
        
        if len(primes_in_chunk) > 0:
            chunk_to_yield = np.empty(len(primes_in_chunk) + 1, dtype=np.uint64)
            chunk_to_yield[0] = last_prime
            chunk_to_yield[1:] = primes_in_chunk
            last_prime = primes_in_chunk[-1]
            yield chunk_to_yield, max_val

def main():
    LIMIT = 10_000_000_000_000 # 10 Trilioni (10^13)
    FILENAME = f"risultato_gpu_{LIMIT}.txt"
    CHECKPOINT_FILE = f"statistiche_congettura_{LIMIT}.npz"
    
    print("=" * 65)
    print(f"  TEST CONGETTURA DI RICKY  —  LIMITE: {LIMIT:,}")
    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        print(f"  Architettura: NVIDIA CUDA CuPy C++ Native  |  GPU: {gpu_name}")
    except:
        print("  Architettura: NVIDIA CUDA via CuPy")

    print(f"  Calcolo analitico di K attivato. Dati salvati in: {CHECKPOINT_FILE}")
    print("=" * 65 + "\n")
    sys.stdout.flush()
    
    t_start = time.perf_counter()
    str_time_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(FILENAME, "w", encoding="utf-8") as f:
        f.write(f"=== TEST CONGETTURA DI RICKY GPU + K STATS ===\n")
        f.write(f"Limite analizzato : {LIMIT:,}\n")
        f.write(f"Inizio test       : {str_time_start}\n")
        f.write(f"Elenco Fallimenti (se presenti):\n")
        f.write(f"--------------------------------------------------\n")
    
    total_pairs = 0
    total_successes = 0
    total_failures = 0
    
    # === Inizializza Contatori Statistiche K ===
    MAX_K = 10000  # un limite safe per le frequenze, si autoadatterà se serve con minlength
    hist_k = np.zeros(MAX_K+1, dtype=np.uint64)
    hist_d = {}
    hist_mod3 = np.zeros((3, MAX_K+1), dtype=np.uint64)
    
    generator = prime_generator(LIMIT)
    threads_per_block = 256
    stream = cp.cuda.Stream()
    
    last_checkpoint_save = 0
    
    try:
        while True:
            try:
                chunk, max_val = next(generator)
            except StopIteration:
                break
                
            p1_arr = chunk[:-1]
            p2_arr = chunk[1:]
            pairs_count = len(p1_arr)
            
            if pairs_count == 0:
                continue
                
            d_p1 = cp.asarray(p1_arr)
            d_p2 = cp.asarray(p2_arr)
            d_k = cp.empty(pairs_count, dtype=cp.uint16)
            
            blocks_per_grid = (pairs_count + threads_per_block - 1) // threads_per_block
            
            with stream:
                analyze_kernel((blocks_per_grid,), (threads_per_block,), (d_p1, d_p2, d_k, pairs_count))
                
            stream.synchronize()
            
            k_vals = d_k.get() 
            
            total_pairs += pairs_count
            
            # Quanti K sono maggiori di 0? (i successi)
            # Quelli uguali a 0 sono i fallimenti
            success_mask = (k_vals > 0)
            successi_blocco = np.count_nonzero(success_mask)
            total_successes += successi_blocco
            
            # --- ACCUMULO STATISTICHE PYTHON ---
            # 1. Calcola d e p1_mod3
            d_vals = (p2_arr - p1_arr).astype(np.uint64) # distanza (sempre <= limit)
            p1_mod3 = (p1_arr % 3).astype(np.uint8)
            
            # Gestiamo dinamicamente eventuali valori folli di k espandendo l'array
            max_k_in_chunk = int(np.max(k_vals)) if len(k_vals) > 0 else 0
            if max_k_in_chunk > MAX_K:
                diff = max_k_in_chunk - MAX_K
                hist_k = np.pad(hist_k, (0, diff))
                hist_mod3 = np.pad(hist_mod3, ((0,0), (0, diff)))
                for key_d in hist_d:
                    hist_d[key_d] = np.pad(hist_d[key_d], (0, diff))
                MAX_K = max_k_in_chunk
                
            # 2. Aggiorna istogramma k totale
            count_k_chunk = np.bincount(k_vals, minlength=MAX_K+1).astype(np.uint64)
            hist_k[:len(count_k_chunk)] += count_k_chunk
            
            # 3. Aggiorna per ogni d (raggruppa)
            unique_d = np.unique(d_vals)
            for d in unique_d:
                mask = (d_vals == d)
                if d not in hist_d:
                    hist_d[d] = np.zeros(MAX_K+1, dtype=np.uint64)
                
                count_d_chunk = np.bincount(k_vals[mask], minlength=MAX_K+1).astype(np.uint64)
                hist_d[d][:len(count_d_chunk)] += count_d_chunk
            
            # 4. Aggiorna per mod 3
            for r in range(3):
                mask = (p1_mod3 == r)
                count_mod3_chunk = np.bincount(k_vals[mask], minlength=MAX_K+1).astype(np.uint64)
                hist_mod3[r][:len(count_mod3_chunk)] += count_mod3_chunk
            
            # --- GESTIONE FALLIMENTI ---
            if successi_blocco < pairs_count:
                fallimenti_idx = np.where(k_vals == 0)[0]
                total_failures += len(fallimenti_idx)
                
                with open(FILENAME, "a", encoding="utf-8") as file:
                    for idx in fallimenti_idx:
                        p1 = p1_arr[idx]
                        p2 = p2_arr[idx]
                        d = p2 - p1
                        M = d * p1
                        lnM = math.log(M) if M > 0 else 0
                        N = int(math.ceil(lnM * lnM))
                        msg = f"⚠ FALLIMENTO: p1={p1}, p2={p2}, d={d}, M={M}, N={N}\n"
                        print(f"\n{msg}", end="")
                        file.write(msg)
            
            elapsed = time.perf_counter() - t_start
            pct = (max_val / LIMIT) * 100
            print(f"\rProcessato fino a: {max_val:,} ({pct:.3f}%)  |  Coppie: {total_pairs:,}  |  Sec: {elapsed:.1f}", end="")
            sys.stdout.flush()
            
            # --- SALVATAGGIO CHECKPOINT STATISTICHE ---
            # Salva ogni ~100 Milioni di numeri (o ~5 milioni di coppie) per evitare I/O blocking freq
            if total_pairs - last_checkpoint_save >= 5_000_000:
                np.savez(CHECKPOINT_FILE, hist_k=hist_k, hist_mod3=hist_mod3, hist_d=hist_d, max_k=MAX_K)
                last_checkpoint_save = total_pairs
            
            # Stop condition: 15 minutes max
            if elapsed >= 14 * 60 + 50:
                print("\n\n[!] Raggiunto il limite di tempo massimo imposto (~15 minuti).")
                with open(FILENAME, "a", encoding="utf-8") as file:
                    file.write("\n[!] Elaborazione fermata per limite di tempo (~15 minuti).\n")
                break    
                
    except KeyboardInterrupt:
        str_msg = "\n\n[!] Elaborazione interrotta manualmente dall'utente."
        print(str_msg)
        with open(FILENAME, "a", encoding="utf-8") as f:
            f.write(str_msg + "\n")
            
    # --- END & SAVE FINALE ---
    # Un ultimo salvataggio per aver tutto perfetto
    try:
        np.savez(CHECKPOINT_FILE, hist_k=hist_k, hist_mod3=hist_mod3, hist_d=hist_d, max_k=MAX_K)
        print(f"\n[+] Statistiche istogrammi di K e D salvate con successo in {CHECKPOINT_FILE}")
    except Exception as e:
        print(f"\n[!] Impossibile salvare statistiche finali: {e}")
        
    t_end = time.perf_counter() - t_start
    str_time_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "=" * 65)
    print(f"  RISULTATI FINALI (GPU NVIDIA RTX)")
    print("=" * 65)
    print(f"  Coppie totali analizzate: {total_pairs:,}")
    print(f"  Successi:                 {total_successes:,}")
    print(f"  Fallimenti:               {total_failures:,}")
    print(f"  Max K trovato:            {MAX_K}")
    print(f"  Tempo totale impiegato:   {t_end:.2f} secondi")
    print("=" * 65 + "\n")
    
    with open(FILENAME, "a", encoding="utf-8") as f:
        f.write(f"\n--------------------------------------------------\n")
        f.write(f"RISULTATO FINALE:\n")
        f.write(f"Fine test                 : {str_time_end}\n")
        f.write(f"Coppie totali analizzate  : {total_pairs:,}\n")
        f.write(f"Esiti positivi            : {total_successes:,}\n")
        f.write(f"Totale fallimenti riscontr: {total_failures:,}\n")
        f.write(f"Max K riscontrato global. : {MAX_K}\n")
        f.write(f"Tempo totale impiegato    : {t_end:.2f} secondi\n")
        if total_failures == 0:
            f.write(f"CONCLUSIONE: Nessun fallimento.\n")

if __name__ == "__main__":
    main()
