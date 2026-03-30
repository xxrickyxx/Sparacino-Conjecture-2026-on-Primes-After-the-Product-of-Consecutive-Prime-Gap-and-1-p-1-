"""
VERIFICA: i bound analitici con s ottimale matchano i conteggi esatti?
"""
import math
from sympy import primerange, isprime, nextprime

GAMMA = 0.5772156649015328606
EG = math.exp(-GAMMA)

def f_linear(s):
    if s <= 2: return 0.0
    if s <= 4: return 2*math.exp(GAMMA)*math.log(s-1)/s
    return 1.0 - 4*math.exp(GAMMA)*math.exp(-(s-2))/(s*(s-1))

def compute_bounds(lnM, s):
    N = lnM**2
    lnN = 2*math.log(lnM)
    ln_z = lnN / s
    z = math.exp(ln_z)
    Wz = EG / ln_z if ln_z > 0.1 else 0.5
    S_lower = N * Wz * f_linear(s) - 0.0486 * N
    if lnM > ln_z and ln_z > 0:
        integral_val = (1.0/lnM) * math.log((lnM - ln_z)/ln_z)
    else:
        integral_val = 0
    P2_upper = 2 * N * max(0, integral_val)
    P3_upper = N * max(0, integral_val)**2
    return S_lower, P2_upper, P3_upper, z

print("=" * 80)
print("  VERIFICA CRITICA: bound analitici vs conteggi esatti")
print("=" * 80)
print()

p = 97
print(f"  {'Coppia':^14} {'lnM':>5} {'s':>4} {'z':>5} "
      f"{'Surv_ex':>8} {'S_low':>7} "
      f"{'P2_ex':>6} {'P2_up':>7} "
      f"{'Pr_ex':>6} {'Pr_low':>7} {'OK':>4}")
print("  " + "-" * 85)

for _ in range(30):
    p2 = nextprime(p)
    d = p2 - p
    M = d * p
    lnM = math.log(M)
    N = max(3, int(math.ceil(lnM**2)))
    
    # Trova s ottimale
    best_s, best_margin = 3.0, -1e18
    for s10 in range(21, 200):
        s = s10/10.0
        r = compute_bounds(lnM, s)
        if r[3] < 2.01: continue
        m = r[0] - r[1] - r[2]
        if m > best_margin:
            best_margin = m; best_s = s
    
    S_low, P2_up, P3_up, z = compute_bounds(lnM, best_s)
    z_int = max(2, int(z))
    
    # Conteggi esatti con z effettivo
    primes_z = list(primerange(2, z_int+1))
    surv, pr_ex, p2_ex, p3_ex = 0, 0, 0, 0
    for k in range(1, N+1):
        val = M + k
        if any(val % pp == 0 for pp in primes_z):
            continue
        surv += 1
        if isprime(val):
            pr_ex += 1
        else:
            temp, nf = val, 0
            f = 2
            while f*f <= temp:
                while temp % f == 0: temp //= f; nf += 1
                f += 1
            if temp > 1: nf += 1
            if nf == 2: p2_ex += 1
            else: p3_ex += 1
    
    pr_low = S_low - P2_up - P3_up
    ok = "SI" if pr_ex > 0 and pr_low > 0 else ("ok" if pr_ex > 0 else "NO")
    
    pair = f"({p},{p2})"
    print(f"  {pair:^14} {lnM:>5.1f} {best_s:>4.1f} {z:>5.1f} "
          f"{surv:>8} {S_low:>7.1f} "
          f"{p2_ex:>6} {P2_up:>7.1f} "
          f"{pr_ex:>6} {pr_low:>7.1f} {ok:>4}")
    p = p2

print()
print("  LEGENDA:")
print("  Surv_ex = sopravvissuti esatti | S_low = lower bound sieve")
print("  P2_ex = semiprimi esatti | P2_up = upper bound analitico")
print("  Pr_ex = primi esatti | Pr_low = S_low - P2_up - P3_up")
print("  OK: SI se Pr_low > 0 (dimostrato), ok se Pr_ex>0 ma non dimostrato")
print()
print("  Se P2_up > P2_ex per OGNI riga: il bound analitico e' VALIDO.")
print("  Se Pr_low > 0 per lnM >= 10: la dimostrazione CHIUDE.")
print()
print("  Firma — " + __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'))
