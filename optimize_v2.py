"""
VERSIONE CORRETTA — ottimizzazione parametro s con calcoli stabili
"""
import math
from sympy import primerange, factorint

GAMMA = 0.5772156649015328606
EG = math.exp(-GAMMA)

def f_linear(s):
    if s <= 2: return 0.0
    if s <= 4: return 2*math.exp(GAMMA)*math.log(s-1)/s
    return 1.0 - 4*math.exp(GAMMA)*math.exp(-(s-2))/(s*(s-1))

def dickman_rho(u):
    if u <= 1: return 1.0
    if u <= 2: return 1.0 - math.log(u)
    h = 0.01
    n = int(u/h) + 2
    rv = [0.0]*(n+1)
    for i in range(n+1):
        t = i*h
        if t <= 1: rv[i] = 1.0
        elif t <= 2: rv[i] = 1.0 - math.log(t)
    for i in range(int(2/h)+1, n+1):
        t = i*h
        j = int((t-1)/h)
        if j < len(rv):
            rv[i] = rv[i-1] - h*rv[j]/t
    return rv[min(int(u/h), n)]

def compute_margin(lnM, s):
    """Calcola S_lower - P2_upper - P3_upper per parametro s."""
    N = lnM**2
    lnN = 2*math.log(lnM)
    
    # z = exp(lnN/s)
    ln_z = lnN / s
    z = math.exp(ln_z)
    if z < 2.01:
        return None  # z troppo piccolo, sieve invalido
    
    # W(z) con Mertens
    Wz = EG / ln_z
    
    # S_lower = N*Wz*f(s) - rho(s)*N
    fval = f_linear(s)
    rho_val = dickman_rho(s)
    S_lower = N * Wz * fval - rho_val * N
    
    # P2 upper: integrale con limiti corretti
    # ∫_{ln_z}^{lnM/2} du/(u*(lnM-u)) = (1/lnM)*ln((lnM/2 - u)/(lnM-u) * u/(lnM/2)) 
    # = (1/lnM) * [ln(u) - ln(lnM-u)] da ln_z a lnM/2
    # = (1/lnM) * (0 - ln(ln_z) + ln(lnM - ln_z))
    # = (1/lnM) * ln((lnM - ln_z)/ln_z)
    if lnM > ln_z and ln_z > 0:
        integral_val = (1.0/lnM) * math.log((lnM - ln_z)/ln_z)
    else:
        integral_val = 0
    
    P2_upper = 2 * N * max(0, integral_val)  # fattore 2 = Selberg upper
    
    # P3 upper: ~N * integral^2 (secondo ordine)
    P3_upper = N * max(0, integral_val)**2
    
    margin = S_lower - P2_upper - P3_upper
    return margin, S_lower, P2_upper, P3_upper, z

print("=" * 70)
print("  OTTIMIZZAZIONE s CON CALCOLI CORRETTI")
print("=" * 70)
print()
print(f"  {'lnM':>6} {'s_opt':>6} {'z':>8} {'S_low':>8} {'P2_up':>8} {'P3_up':>8} {'Margin':>9} {'OK':>4}")
print("  " + "-" * 62)

first_ok = None
for lnM in list(range(10, 80)) + list(range(80, 501, 10)):
    best = None
    for s10 in range(21, 300):  # s da 2.1 a 30
        s = s10 / 10.0
        result = compute_margin(lnM, s)
        if result is None: continue
        margin = result[0]
        if best is None or margin > best[0]:
            best = (margin, s, result)
    
    if best is None: continue
    margin, s_opt, (_, S_low, P2_up, P3_up, z) = best
    ok = "SI" if margin > 0 else "no"
    if margin > 0 and first_ok is None:
        first_ok = lnM
    
    show = lnM <= 50 or lnM % 20 == 0
    if first_ok and abs(lnM - first_ok) <= 5: show = True
    if show:
        print(f"  {lnM:>6} {s_opt:>6.1f} {z:>8.1f} {S_low:>8.1f} {P2_up:>8.2f} {P3_up:>8.2f} {margin:>9.1f} {ok:>4}")

print()
if first_ok:
    print(f"  *** SOGLIA OTTIMALE: lnM >= {first_ok} ***")
    print(f"  *** => M >= 10^{first_ok/math.log(10):.1f} ***")
    print()
    
    # Con il fattore C(d) di miglioramento
    print("  Con il fattore strutturale C(d):")
    for d in [2, 4, 6, 12, 30]:
        ff = factorint(d)
        Cd = 1.0
        for q in ff: Cd *= q/(q-1)
        # Il fattore C(d) migliora S_lower => la soglia scende
        # S_lower(d) ≈ C(d) * S_lower => margine migliora
        # Cercando il nuovo lnM dove C(d)*S_low > P2+P3
        new_first = None
        for lnM2 in range(5, 500):
            result = compute_margin(lnM2, 3.0)
            if result is None: continue
            margin_base = result[0]
            # Con C(d): S_lower *= C(d), P2 e P3 invariati
            S_adj = result[1] * Cd
            new_margin = S_adj - result[2] - result[3]
            if new_margin > 0 and new_first is None:
                new_first = lnM2
                break
        if new_first:
            print(f"    d={d:>2}: C(d)={Cd:.1f}, soglia lnM>={new_first} => M>=10^{new_first/math.log(10):.0f}")
    
    # Gap con verifica GPU
    print()
    print(f"  Verifica GPU: M <= 10^31")
    print(f"  Soglia generica: M >= 10^{first_ok/math.log(10):.0f}")
    gap = first_ok/math.log(10) - 31
    if gap <= 0:
        print("  >>> NESSUN GAP! DIMOSTRAZIONE COMPLETA! <<<")
    else:
        print(f"  Gap: {gap:.0f} ordini di grandezza")
        print()
        print("  OPZIONE A: estendere GPU (impraticabile per gap > 10)")
        print("  OPZIONE B: usare Selberg upper sieve ESPLICITO sui P2")
        print("             con i termini di resto controllati da BV")

print()
print("  Firma — " + __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'))
