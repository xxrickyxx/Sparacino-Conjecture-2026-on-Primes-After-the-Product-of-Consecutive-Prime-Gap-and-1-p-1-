"""
APPROCCIO NUOVO: LA STRUTTURA DI M = d*p1 BYPASSA LA PARITY BARRIER?
======================================================================
L'utente ha ragione: Sparacino NON chiede primo in (x, x+ln^2(x)] per
TUTTI gli x — solo per x = d*p1 dove (p1,p2) e' una coppia consecutiva.

Questa specialita' potrebbe permettere argomenti che falliscono per Cramer.

Esploriamo TRE approcci nuovi:

A) Metodo delle somme di von Mangoldt per M specifico
B) Bound sulla distanza al primo successivo usando p_1 come modulo
C) Confronto empirico: densita' di primi intorno a M vs. numeri generici
"""
import math
import sys
from sympy import isprime, nextprime, primerange, factorint

GAMMA = 0.5772156649015328606

# ================================================================
# APPROCCIO A: La somma di von Mangoldt S(M,N) = sum Λ(M+k)
# ================================================================
print("=" * 65)
print("APPROCCIO A: SOMME DI VON MANGOLDT PER M = d*p1")
print("=" * 65)
print()
print("  S(M,N) = #{primi in (M, M+N]} * ln(primo) (approx.)")
print("  Per PNT: S(M,N) ~ N (valore atteso)")
print("  Vogliamo mostrare S(M,N) > 0 SEMPRE per Sparacino M.")
print()

def S_vonM(M, N):
    """Conta i primi in (M, M+N] e ritorna la 'somma di von Mangoldt'."""
    primes = [k for k in range(1, N+1) if isprime(M+k)]
    return len(primes), sum(math.log(M+k) for k in primes)

def prime_density_M(M, N):
    """Densita' effettiva di primi in (M, M+N] vs. attesa da PNT."""
    actual = sum(1 for k in range(1, N+1) if isprime(M+k))
    expected = N / math.log(M)
    return actual, expected, actual / expected if expected > 0 else 0

# Test su coppie Sparacino
print(f"  {'Coppia':^15} {'M':>10} {'N':>5} {'Primi':>6} {'Attesi':>8} {'Ratio':>8} {'k_min':>6} {'k_max':>6}")
print("  " + "-" * 72)

p = 3
sparacino_ratios = []

for _ in range(30):
    p2 = nextprime(p)
    d = p2 - p
    M = d * p
    N = int(math.ceil(math.log(M)**2))
    
    actual, expected, ratio = prime_density_M(M, N)
    
    # Trova k_min e k_max
    primes_k = [k for k in range(1, N+1) if isprime(M+k)]
    k_min = primes_k[0] if primes_k else -1
    k_max = primes_k[-1] if primes_k else -1
    
    sparacino_ratios.append(ratio)
    
    pair = f"({p},{p2})"
    print(f"  {pair:^15} {M:>10} {N:>5} {actual:>6} {expected:>8.1f} {ratio:>8.3f} {k_min:>6} {k_max:>6}")
    p = p2

print()
print(f"  Ratio medio Sparacino: {sum(sparacino_ratios)/len(sparacino_ratios):.3f}")
print(f"  Ratio min Sparacino:   {min(sparacino_ratios):.3f}")
print()

# Confronto con numeri generici della stessa dimensione
print("  CONFRONTO con interi generici (stessa dimensione di M):")
print(f"  {'M (generico)':>15} {'N':>5} {'Primi':>6} {'Attesi':>8} {'Ratio':>8}")
print("  " + "-" * 50)

import random
random.seed(42)
p = 3
generic_ratios = []
for _ in range(15):
    p2 = nextprime(p)
    d = p2 - p
    M_spark = d * p
    N = int(math.ceil(math.log(M_spark)**2))
    
    # Usa un M generico della stessa dimensione ma non di Sparacino
    # (un numero casuale vicino a M_spark)
    offset = max(N+10, 3*N + 1)
    M_gen = M_spark + random.randint(N+2, offset)  # fuori dalla finestra di Sparacino
    actual_g, expected_g, ratio_g = prime_density_M(M_gen, N)
    generic_ratios.append(ratio_g)
    
    print(f"  {M_gen:>15} {N:>5} {actual_g:>6} {expected_g:>8.1f} {ratio_g:>8.3f}")
    p = p2

print()
print(f"  Ratio medio generico: {sum(generic_ratios)/len(generic_ratios):.3f}")
print()
print("  OSSERVAZIONE: il ratio e' simile. La struttura M=d*p1 NON")
print("  sembra dare una densita' di primi sistematicamente piu' alta.")
print("  Il vantaggio e' nell'ALBERO DI SIEVE (lemma C(d)), non nella densita' grezza.")
print()

# ================================================================
# APPROCCIO B: USARE p1 COME MODULO — argomento di RESIDUI
# ================================================================
print("=" * 65)
print("APPROCCIO B: ARGOMENTO DI RESIDUI MODULARI")
print("=" * 65)
print()
print("  IDEA NUOVA: M = d*p1, e p1 > ln^2(M) = N.")
print("  Consideriamo i k in {1,...,N} modulo p1:")
print("  M+k ≡ k (mod p1) per Lemma 2.")
print()
print("  Questo significa che {M+1,...,M+N} forma una sezione COMPLETA")
print("  del sistema di residui mod p1 (poiche' N < p1).")
print()
print("  PER DIRICHLET: in ogni classe di residui r (mod p1) con gcd(r,p1)=1,")
print("  la densita' di primi e' 1/phi(p1) = 1/(p1-1).")
print()
print("  Ora: il numero di k in {1,...,N} con gcd(k, p1) = 1 e' semplicemente")
print("  N - 0 = N (poiche' p1 > N => la sola classe con gcd > 1 sarebbe k=0 mod p1,")  
print("  ma k in {1,...,N} < p1 => nessun k e' divisibile da p1).")
print()
print("  Quindi tutti i k in {1,...,N} sono in classi di residui AMMISSIBILI mod p1.")
print()
print("  CONSEGUENZA FORMALE:")
print("  Nella forma aritmetica di Dirichlet applicata a p in S ≡ k (mod p1):")
print("  pi(M+N; p1, k) - pi(M; p1, k) ~ N/(p1 * ln(M/p1))")
print()

p = 97
print(f"  {'Coppia':^12} {'p1':>8} {'N':>5} {'N/p1':>8} {'Dirichlet ok':>14}")
print("  " + "-" * 55)
for _ in range(10):
    p2 = nextprime(p)
    d = p2 - p
    M = d * p
    N = int(math.ceil(math.log(M)**2))
    print(f"  ({p},{p2}) {p:>8} {N:>5} {N/p:>8.4f}  {'N<p1 ✓' if N < p else 'N>=p1 ✗':>14}")
    p = p2

print()
print("  PROBLEMA: Dirichlet su AP (k mod p1) non conta DIRETTAMENTE")
print("  i primi in (M, M+N]. Ci dice la densita' ASINTOTICA per p1 fisso,")
print("  ma non la densita' in questo specifico interval di lunghezza N << p1.")
print()

# ================================================================
# APPROCCIO C: NUOVO — Usare p2 per costruire un primo in (M, M+N]
# ================================================================
print("=" * 65)
print("APPROCCIO C: NUOVO — p2 = p1+d COME 'ANCORA' PER TROVARE PRIMI")
print("=" * 65)
print()
print("  Sappiamo che p2 = p1 + d esiste ed e' primo.")
print("  M = d * p1 = p2*p1 - p1^2.")
print()
print("  IDEA: possiamo esprimere numeri vicino a M in termini di p1 e p2?")
print()
print("  M + k = p1*p2 - p1^2 + k")
print()
print("  Per k = p1*(p1-p2) + s = -p1*d + s  => non semplifica.")
print()
print("  Proviamo k* = (p2^2 - p1^2)/... nope.")
print()
print("  APPROCCIO DIVERSO: Sia q il primo piu' piccolo > M.")
print("  Per Bertrand: q < 2M.")
print("  Per la congettura: q <= M + ln^2(M).")
print("  Per Sparacino M = d*p1: possiamo dire qualcosa di piu' su q?")
print()

p = 97
print(f"  {'Coppia':^12} {'M':>8} {'N':>5} {'q_next':>12} {'q_next-M':>10} {'<N?':>6} {'ln^2M':>8}")
print("  " + "-" * 65)
for _ in range(15):
    p2 = nextprime(p)
    d = p2 - p
    M = d * p
    N = int(math.ceil(math.log(M)**2))
    q = nextprime(M)
    gap = q - M
    ok = "SI ✓" if gap <= N else "NO ✗"
    print(f"  ({p},{p2}) {M:>8} {N:>5} {q:>12} {gap:>10} {ok:>6} {N:>8}")
    p = p2

print()

# ================================================================
# APPROCCIO D: NUOVO — FORMULA DIRETTA USANDO LA STRUTTURA MOLTIPLICATIVA
# ================================================================
print("=" * 65)
print("APPROCCIO D: LA STRUTTURA d < p1 => p1 > sqrt(M)")
print("(QUESTO E' IL GENUINO VANTAGGIO STRUTTURALE)")
print("=" * 65)
print()
print("  Per Bertrand: d = p2 - p1 < p1 (poiche' p2 < 2*p1 per Bertrand).")
print("  Quindi: M = d*p1 < p1^2 => p1 > sqrt(M).")
print()
print("  Questo e' CRUCIALE:")
print("  - Per l'intervallo (M, M+N], qualsiasi fattore primo")
print("    q <= sqrt(M+N) < sqrt(M^{1+eps}) soddisfa q < p1.")
print("  - Ma p1 | M => M+k ≡ k (mod p1) per k in {1,...,N}.")
print("  - Poiche' q < p1 e q != p1, q non ha questa proprieta'.")
print()
print("  CONSEGUENZA: nella fattorizzazione dei compositi di (M, M+N],")
print("  il fattore p1 NON PUO' MAI APPARIRE (Lemma 2).")
print("  Questo e' gia' noto, ma ora vediamo il perche' STRUTTURALE:")
print("  p1 > sqrt(M) implica che p1 e' il 'grande fattore'.")
print()
print("  Per i semiprimi s = p*q in (M, M+N] con p <= q:")
print("  - p <= sqrt(M) < p1 => p != p1")
print("  - q = s/p >= M/p > M/sqrt(M) = sqrt(M) > sqrt(N) = ln(M)")
print()
print("  Quindi tutti i semiprimi in (M,M+N] hanno due fattori")
print("  ENTRAMBI diversi da p1, e il fattore maggiore q > ln(M) >> N^{1/2}.")

print()
print("  NUOVO BOUND SUI SEMIPRIMI IN (M, M+N]:")
print("  Per s = p*q in (M, M+N] con p <= q:")
print("  p <= sqrt(s) < sqrt(M + ln^2(M)) = sqrt(M) * sqrt(1 + ln^2(M)/M)")
print()

for lnM in [10, 20, 30, 50, 70]:
    M_approx = math.exp(lnM)
    N_val = int(lnM**2)
    sqrt_M = math.sqrt(M_approx)
    correction = math.sqrt(1 + N_val/M_approx)
    eff_bound = sqrt_M * correction
    print(f"  ln(M)={lnM}: sqrt(M)={sqrt_M:.2e}, bound_p={eff_bound:.2e}, N={N_val}")

print()
print("  Il bound su p nei semiprimi e' sqrt(M) ≈ 10^{lnM/2}.")
print("  Ma noi abbiamo N = ln^2(M) ≈ {lnM}^2 candidati.")
print()
print("  Per quanti p in (N, sqrt(M)) esiste q = (M+k)/p primo?")
print("  Questa e' la DOMANDA CHIAVE (Type II semiprimes).")

# ================================================================
# CONFRONTO FINALE: SPARACINO vs CRAMER — DIFFERENZA STRUTTURALE
# ================================================================
print()
print("=" * 65)
print("CONFRONTO FINALE: SPARACINO vs CRAMER")
print("=" * 65)
print()
print("  Cramer chiede: PER OGNI x, primo in (x, x+ln^2(x)].")
print("  Sparacino chiede: SOLO PER x = d*p1, primo in (d*p1, d*p1+ln^2(d*p1)].")
print()
print("  Differenze strutturali di Sparacino:")
print()
print("  1. [CONFERMATO]  x = d*p1 e' SEMPRE PARI")
print("     => Tutti i k dispari sono candidati (elimina meta' delle ostruzioni)")
print()
print("  2. [CONFERMATO]  p1 | x con p1 > sqrt(x)")
print("     => Nessun M+k ha fattore p1 (eliminata una classe di compositi)")
print()
print("  3. [CONFERMATO]  q | x per q | d (piccoli q)")
print("     => Fattore C(d) > 1 miglioramento locale")
print()
print("  4. [NUOVO] x = d*p1 e d e' un GAP fra primi")
print("     => d e' sempre PARI (e' diff. di 2 primi dispari)")
print("     => d e' sempre EVEN => tutti i fattori di d diversi da 2 sono dispari")
print("     => d non e' mai primo per d > 2 (esclusi twin primes)")
print("     => La struttura di d LIMITA i semprimi vicino a x!")
print()
print("  5. [CRITICO] La somma sum_{Sparacino M} 1/M CONVERGE")
print("     => 'quasi tutti' i punti Sparacino hanno un primo vicino")
print("     => Ma 'quasi tutti' != 'tutti' (Borel-Cantelli e' probabilistico)")
print()
print("  COSA MANCA ANCORA:")
print("  Un argomento che usi la struttura (1)-(4) per dimostrare")
print("  che i semiprimi in (M,M+N] non possono COPRIRE tutti i k ammissibili.")
print()
print("  => Questo richiederebbe mostrare che:")
print("     #{semiprimi in (M,M+N]} < #{candidati in {1,...,N}}")
print("     ovvero |Type II| < |C|")
print()
print("  Questo e' ESATTAMENTE il tipo di argomento della 'Large Sieve'")
print("  o 'Methode des zeros' che viene usato in Chen's Theorem.")
print("  Chen lo ha dimostrato per i numeri PARI n = p + P_2,")
print("  potremmo adattarlo per n = M con la struttura di Sparacino.")
print()
print("  PROSSIMO PASSO CONCRETO:")
print("  Leggere il metodo di Chen (1973) e verificare se si adatta")
print("  all'impostazione M = d*p1 con i vincoli strutturali 1-4.")
print()
print("  Questo e' GENUINAMENTE FATTIBILE come lavoro matematico.")
print()
print("  Firma — " + __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'))
