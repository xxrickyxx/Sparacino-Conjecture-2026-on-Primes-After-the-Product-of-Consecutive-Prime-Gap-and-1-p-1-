# Sparacino Conjecture: Primes After a Consecutive Prime Gap Times Its Lower Prime

Riccardo Sparacino

Version v13 | 25 October 2026 | Research note

DOI reserved for this version: 10.5281/zenodo.23638603

**Status: the universal conjecture remains open.** This version supersedes the mathematical and computational claims in earlier descriptions where they conflict with the corrections stated here. It does not claim a proof or a counterexample for the actual primes.

## Abstract

For consecutive primes p<p+d, set M=dp and N=⌈log²M⌉. The Sparacino conjecture asserts that (M,M+N] contains a prime. We give a reproducible finite verification derived from the exhaustive prime-gap data of Oliveira e Silva, Herzog and Pardi: the assertion holds whenever M<3,999,999,999,999,999,887, and in particular for every consecutive pair with p≤3,533,568,904,593,639. This uses their published exhaustive computation as an external input; the reduction and its rational inequalities are independently checked here. We also prove, using the uniform linear-forms theorem of Maynard, that there are absolute constants a>0 and D for which infinitely many consecutive pairs with one fixed gap d≤D have first-prime distance K≥a log log M. We distinguish these results for actual primes from a theorem predicting failures in an explicitly specified Granville random model. Independent exact sieving and residue-class experiments are supplied with code, data, checksums and limitations. Neither the probabilistic theorem nor the finite verification resolves the universal conjecture.

## 1. Statement and scope

Let p₁<p₂ be consecutive primes, including the first pair (2,3), and define

$$
d=p₂−p₁,     M=dp₁,     N(M)=⌈(log M)²⌉,
$$

$$
K(M)=min{k≥1 : M+k is prime}.
$$

All logarithms are natural. The conjecture is K(M)≤N(M) for every consecutive pair. The interval is open at M and closed at M+N(M).

Three kinds of conclusion must be separated: a finite implication from published exhaustive gap data; unconditional deductions about actual primes from established theorems; and statements inside a random model. Experiments using probable-prime tests are explicitly distinguished from certified primality. No numerical experiment presented here implies a universal bound.

## 2. Corrections to earlier versions

### 2.1 A larger prime-containing interval is insufficient

The Baker–Harman–Pintz result [3] gives primes in intervals on the scale x^0.525 for sufficiently large x. Since log²x is much smaller, finding a prime in the larger interval does not locate a prime in the smaller one. Containment in that direction proves no upper bound of length log²x. Earlier statements asserting that this result completes the Sparacino proof are withdrawn. No effective threshold joining the earlier GPU samples to a universal proof has been established.

### 2.2 Configured limits and completed computations

The archived GPU run contains 139,864,010 pairs and an explicit manual-interruption message. Its completed blocks correspond to π(2,900,000,000)−1 pairs; π(2,900,000,000)=139,864,011. The configured limit 10^13 was not completed. The archived value 451 is a first-prime offset K, not an odd gap between consecutive odd primes. Historical files are retained for provenance, with this correction.

### 2.3 Fixed Miller–Rabin bases at large sizes

The original 128-bit program uses the first fifteen prime bases, from 2 through 47. This is not a deterministic primality test throughout the claimed range. For example,

$$
59,276,361,075,595,573,263,446,330,101
$$

equals 172,157,429,516,701 times 344,314,859,033,401 and passes all fifteen strong tests. This is a counterexample to the asserted testing guarantee, not to the Sparacino conjecture; we do not assert that the experiment encountered this number.

### 2.4 A sieve window is not the size of its candidates

Sieving up to N^(1/3) does not leave numbers with at most three prime factors when the candidates have size M. For the actual consecutive pair (199,211), d=12, M=2388 and N=61, the candidate M+13=2401=7^4 survives that cutoff, since 7>61^(1/3). The relevant factor-count comparison involves log(M+N)/log z, not log N/log z. Earlier sieve arguments omitting this distinction do not prove primality.

The global empirical claim K/N≤0.414 is also withdrawn: exact independent sieving finds K/N=139/211 for the twin pair beginning at p=1,005,371. This ratio is below 1 and is not a counterexample to the conjecture.

## 3. A finite verification from published exhaustive gaps

**Theorem 1 (finite, conditional only on the correctness of the published exhaustive dataset).** For every consecutive pair with

$$
M< P*=3,999,999,999,999,999,887,
$$

the conjectured inequality holds. In particular it holds for every consecutive pair with

$$
p₁≤3,533,568,904,593,639.
$$

The external dataset is the table of all observed gap lengths from the exhaustive scan up to 4×10^18 by Oliveira e Silva, Herzog and Pardi [1]. The table has 696 gap-length rows. For each length g, it supplies the first starting prime P(g) and occurrence count n(g). The local copy and its source attribution are included in the supplement.

### 3.1 From first occurrences to every covered gap

For every row we verify with exact rational arithmetic that

$$
g≤⌈log²P(g)⌉.
$$

Every later occurrence of length g starts at a prime a≥P(g), so monotonicity gives g≤⌈log²a⌉. Consequently this bound applies to every gap represented in the exhaustive scan, without re-enumerating its endpoints locally.

For the logarithms, take the first 32 positive terms of

$$
log 2 = 2 Σ[j≥0] 1 / ((2j+1)3^(2j+1)).
$$

This rational partial sum L₂ is strictly below log 2. With e=⌊log₂P(g)⌋, the inequality (eL₂)²>g−1 suffices for 694 rows. For P(g)=3 and 7, eight positive terms of the analogous expansion 2Σ z^(2j+1)/(2j+1), with z=(P(g)−2^e)/(P(g)+2^e), provide the additional lower bound. Every comparison is exact; the smallest certified margin above g−1 exceeds 0.2069.

### 3.2 Boundary and transfer to an arbitrary integer

The occurrence counts give a telescoping identity:

$$
2 + Σ[g] g n(g) = P* = 4×10^18−113.
$$

The supplement separately certifies primality of P* with a recursive Lucas certificate using the full factorization of P*−1. Its verifier uses integer products, modular exponentiation and gcd; probable-prime tests used during certificate generation are not trusted by the verifier.

For an integer 2≤m<P*, let a be the largest prime at most m and let b be the next prime. Since P* is prime, b≤P*. Thus the gap is covered and

$$
0<b−m≤b−a≤⌈log²a⌉≤⌈log²m⌉.
$$

This proves the theorem for every such m, including each Sparacino point M. The inclusive product bound is therefore P*−1=3,999,999,999,999,999,886.

### 3.3 A uniform bound on the lower input prime

Set L=⌊(P*−1)/1132⌋=3,533,568,904,593,639. For p₁≤L, Bertrand's postulate places p₂<2p₁<4×10^18, so the input gap is covered by the published scan. Its table shows that the first occurrence of any gap larger than 1132 starts at 43,841,547,845,541,059, with gap 1184. Since this is larger than L, all these input gaps satisfy d≤1132. Therefore

$$
M≤1132L=3,999,999,999,999,999,348<P*.
$$

For a fixed input gap d, the product bound also yields p₁≤⌊(P*−1)/d⌋. For twin primes this gives p₁≤1,999,999,999,999,999,943. The published exhaustive scan is an external computational input: it has not been repeated here.

## 4. Elementary structure and counting the starting points

For p>2 and prime endpoints p,p+d, one has

$$
gcd(dp+1, d(d−1)(d+1))=1.
$$

Indeed dp+1 is coprime to d. If a prime q divided both dp+1 and d²−1, the identity dp+1=d(p+d)−(d²−1) would force q=p+d. But p+d>d+1, so it cannot divide d−1 or d+1. Consecutivity is not needed for this lemma. The first pair (2,3), where d=1, is excluded. The lemma does not establish primality: with (p,p+d)=(109,113), dp+1=437=19×23.

Let S={pₙ(pₙ₊₁−pₙ)} and A(X)=#(S∩[1,X]). These points are distinct: Bertrand gives dₙ<pₙ for pₙ>2, making pₙ the largest prime factor of Mₙ. There are absolute positive constants c,C such that, for large X,

$$
cX/log²X ≤ A(X) ≤ CX log log X/log²X.
$$

For the upper bound, points with p≤X/log X contribute O(X/log²X). Otherwise d<log X. The uniform Brun prime-pair upper bound, in the form recalled by Erdős and Nathanson [4], bounds the remaining contribution by

$$
O(X/log²X) Σ[d≤log X] F(d)/d,
$$

where F(d)=∏[q|d](1+1/q). Using F(d)≤Σ[a|d]1/a, the sum is at most ζ(2)(1+log log X). For the lower bound, PNT and telescoping show that at least a fixed positive fraction of primes in (Y,2Y] have next gap at most 4 log Y. Their products are at most 8Y log Y. Taking Y=X/(8 log X) proves the lower bound.

Partial summation gives Σ[M∈S]1/M<∞ and a tail O((log log Y+1)/log Y). This convergent series does not rule out an infinite exceptional subset of actual primes.

The factor ∏[q|d]q/(q−1) should not be interpreted as an unconditional density gain across every offset: over a complete cycle modulo q|d, exactly one in q offsets is still divisible by q. Local residue information is useful, but it does not supply the missing uniform upper bound.

## 5. Unconditional long composite stretches after actual starting points

**Theorem 2.** There are absolute constants a>0 and D such that, for every sufficiently large X, there is a consecutive pair p,p+d with X≤p≤3X, 2≤d≤D, for which all numbers

$$
dp+1, …, dp+⌊a log log(dp)⌋
$$

are composite. Moreover a single fixed value d*≤D occurs in infinitely many such pairs. The constants are not made numerically explicit. This is a deduction from Maynard's theorem; no claim of bibliographic priority is made.

### 5.1 Fixing an admissible tuple and forcing consecutivity

Use Maynard's Theorem 3.4 [2] with m=2 and ε=1/100, and fix a sufficient tuple size k. Let R be the product of primes at most k and take the admissible even tuple H={h₁,…,h_k} with h_i=(i−1)R, so 0=h₁<…<h_k=D. In particular every difference d=h_j−h_i is at least 2.

Write B=A+Qn. For every even j∈[0,D] outside H, choose a distinct prime r_j>D and impose B≡−j (mod r_j). Also require B odd. Thus every integer in B+[0,D] outside the tuple is composite for large B. Select a fixed C greater than D, 2k+2 and every r_j. For the remaining primes at most C choose residues of B avoiding all −h_i, using admissibility. The prescribed r_j residues also avoid all these roots because r_j>D and j is not a tuple member.

### 5.2 Presieving the target intervals

Let the desired composite length H tend to infinity, with H>C+D². For each prime C<q≤H, independently choose B modulo q uniformly among its q−k allowed residues. Consider every tuple pair h_i<h_j, put d=h_j−h_i and consider targets d(B+h_i)+s, 1≤s≤H.

If gcd(s,d)>1, the target is already composite. Otherwise define

$$
F(s)=∏[1≤ℓ≤k] (s−d(h_ℓ−h_i)).
$$

Every factor is nonzero and has absolute value at most H+D². For q>C the target has one root in B modulo q; it coincides with a forbidden endpoint root exactly when q divides F(s). Its survival probability is therefore

$$
∏[C<q≤H](1−1/(q−k)) · ∏[q|F(s), C<q≤H](q−k)/(q−k−1).
$$

The first product is O_k(1/log H), by Mertens and a convergent product comparison. Since q>2k+2, each factor in the second product is at most (q/(q−1))². Hence that product is at most

$$
∏[1≤ℓ≤k] (|s−d(h_ℓ−h_i)| / φ(|s−d(h_ℓ−h_i)|))².
$$

For every fixed positive integer r there is a finite constant C_r such that

$$
Σ[m≤T] (m/φ(m))^r ≤ C_r T.
$$

To prove this, expand (m/φ(m))^r=Σ[v|m] a_r(v), where a_r is nonnegative, multiplicative, supported on squarefree v, and a_r(q)=(q/(q−1))^r−1. The mean is at most TΣ a_r(v)/v; its Euler product converges because a_r(q)/q=O_r(q^(−2)).

Hölder with k factors, using r=2k, now bounds the sum in s of the preceding product by 2C_(2k)(H+D²)≤4C_(2k)H. Each positive integer occurs at most twice as the absolute value of a shifted s. Summing over the fixed number of tuple pairs, the expected number of surviving coprime targets is at most C₁H/log H. Therefore some choice of the presieve residues leaves at most that many targets. Fixed residues at primes at most C can only remove additional targets.

### 5.3 Completing the covering with a small modulus

For each surviving target choose a new distinct prime q>H+D² and impose

$$
B≡−h_i−s d^(−1) (mod q).
$$

This kills the target while protecting every B+h_ℓ. Otherwise q would divide the nonzero integer s−d(h_ℓ−h_i), whose absolute value is smaller than q. PNT supplies enough distinct primes between H+D² and C₂H, for a fixed sufficiently large C₂.

The resulting CRT modulus includes all primes up to H and these additional primes, so for a fixed C₀ and all large H,

$$
log Q≤θ(H)+(C₁H/log H) log(C₂H)≤C₀H.
$$

This is the improvement over assigning a fresh prime to every offset: the presieve reduces the modulus estimate from log Q=O(H log H) to O(H).

### 5.4 Applying the uniform Maynard theorem

Given X large, take H=⌊log log X/(4C₀)⌋, so Q≤(log X)^(1/4), and put T=⌈X/Q⌉. The forms Qn+A+h_i are admissible: for primes dividing Q they are protected by construction; for all other primes use the bijection n↦Qn+A modulo that prime and the original tuple's admissibility.

Their leading coefficients satisfy Q≤log T for large X, and their constant terms are at most Q+D≪T. Maynard's Theorem 3.4 applies uniformly to these forms on [T,2T]; its size threshold does not depend on the particular forms within its stated coefficient bounds. It gives at least two prime forms for ≫T/(log T)^k integers n.

Select adjacent primes among the tuple's prime values. All positions outside the tuple were forced composite, so these primes are globally consecutive. Their lower endpoint lies in [X,3X], their gap is at most D, and every target offset up to H is composite. Since log log(dp)=log log X+o(1), taking, for example, a=1/(8C₀) proves the assertion.

For large H, Q>D; intervals belonging to distinct n are disjoint. One also obtains ≫X/(log X)^(k+1) such pairs in [X,3X]. Finally take X=4^j and use the finite number of possible d values to obtain one fixed d* infinitely often. This proves Theorem 2.

Theorem 2 is compatible with K≤⌈log²M⌉: its lower bound grows much more slowly. It does not show infinitely many twin primes.

## 6. Random models: a distinction at the critical scale

### 6.1 Cramér on a fixed set of real starting points

Fix the actual set S and, independently of it, label each integer m≥3 with probability 1/log m. The probability of an empty window of length ⌈c log²M⌉ after M is exp(−c/2+o(1))M^(−c). Because Σ[M∈S]1/M converges, the first Borel–Cantelli lemma gives only finitely many empty windows almost surely when c≥1. It allows finite exceptions and is not a theorem about the actual primes.

It is incorrect to claim uniform convergence of Cramér counts divided by log x to 1 for windows of exactly log²x. The uniform law quoted in [5, §2.5] requires y/log²x→∞. At the critical length, elementary binomial large-deviation estimates actually give almost surely liminf Z(x)/log x=0 and limsup Z(x)/log x=e, with x ranging over integers. The rate is I(α)=1−α+α log α; independent disjoint windows and Borel–Cantelli prove the lower assertions when I(α)<1, and a union bound proves the upper assertion when I(α)>1. A detailed proof is supplied in the accompanying research note. For actual primes, Maier's theorem gives anomalous intervals of length (log x)^A for fixed A>1. This is a separate reason not to transfer a uniform law of the random model to the primes; the exact log² scale must be treated separately.

### 6.2 An explicit Granville model

For I_j=(2^j,2^(j+1)], j sufficiently large, put

$$
A_j=log(2^j)/log log(2^j),    Q_j=∏[q≤A_j]q,
$$

and Θ_j=∏[q≤A_j](1−1/q). Exclude integers not coprime with Q_j and include every remaining integer m∈I_j, independently over all m, with probability 1/(Θ_j log m). Initial blocks can be defined arbitrarily. This is a specified instance of the Granville model discussed in [5]. The condition A_j=o(log(2^j)) is explicit; it does not follow merely from writing A_j=log^(1−o(1))(2^j).

**Theorem 3 (only in this random model).** For every 0<c<2e^(−γ), almost surely infinitely many consecutive model elements n,n+2 have no model element in (2n,2n+⌈c log²(2n)⌉]. In particular the model analogue of the Sparacino conjecture fails infinitely often when c=1.

*Proof.* Put X=2^j and L=log X. Choose n in (X/2,X] away from O(L²) boundary positions, with n≡5 (mod 6) and 2n≡−1 (mod q) for 5≤q≤A_j. The CRT class protects n,n+2 through A_j and hence through A_(j−1). Since log Q_j=(1+o(1))A_j=o(L), there are X^(1−o(1)) candidates. Also Q_j≫L², making their target windows disjoint.

The free offsets h in a window of length y are precisely the odd h with h≢2 (mod 3) and with h−1 avoiding every prime from 5 through A_j. Buchstab's rough-number asymptotic, since log y/log A_j→2, gives (1/2+o(1))y/log A_j survivors. To check the parity factor, write h−1=2^a3^b r with a≥1. The total weight from b≥1 is 1/2; when b=0, half the rough r satisfy the required residue modulo 3, giving the other 1/2. The summable tail in a,b justifies passage to the asymptotic.

Mertens gives inclusion probability (e^γ+o(1))log A_j/L at each free position. Thus an empty window has probability X^(−ce^γ/2+o(1)); the quadratic term in the logarithm of this product is O(log A_j)=o(L). The two input elements are included with probability X^(−o(1)). Their labels lie in I_(j−1), whereas all target labels lie in I_j.

For distinct candidates all relevant labels are disjoint: input pairs and target windows occupy different blocks, and the spacing Q_j exceeds the window lengths. The expected number of desired events is μ_j=X^(1−ce^γ/2+o(1)). For c<2e^(−γ) this tends to infinity as a positive power of X. The probability of no event is at most exp(−μ_j), summable in j. The first Borel–Cantelli lemma suffices; independence between successive blocks is unnecessary. Since n+1 is even and excluded, n,n+2 are consecutive model elements. This proves Theorem 3.

For c=1 the exponent is 1−e^γ/2=0.10946… . The theorem is a statement about a random set, not a constructed counterexample for the actual primes.

### 6.3 Limits of conditional and numerical extrapolations

The exploratory Hardy–Littlewood moment hypothesis for {n,n+2,2n+h} uses growing tuple size and an absolute error O(x^(1−c)); it is stronger than the usual conjecture for each fixed tuple. The proposed transfer through Bonferroni only reaches empty windows c′log²x for c′<2ce^(−γ). Crossing constant 1 requires c>e^γ/2≈0.8905; c<1/2 reaches only c′<e^(−γ)≈0.56146. No such transfer hypothesis is assumed in Theorems 1 or 2.

An exploratory numerical crossing near log M≈3.5×10^7, and statements assigning a 5–20% chance to isolated failures, are not certified thresholds or probabilities for actual primes. Their values depend on the model, cutoff, class family and numerical approximations. In a model with almost surely infinitely many failures, the probability of at least one failure beyond any fixed finite bound is 1; a smaller probability must refer to a different model or a specified bounded range. No claim that a first real counterexample needs fifteen million digits is made here.

## 7. Independent computations and residue-class experiments

### 7.1 Exact small-range verification

An independent segmented Eratosthenes sieve checks every consecutive pair with p₁≤10^7: 664,579 pairs, no failures, maximum product 1,280,030,152 and maximum observed offset 235. Apart from the initial pair (2,3), the largest K/N is 139/211≈0.658768 at (p₁,p₂)=(1,005,371,1,005,373), with M=2,010,742 and next prime 2,010,881. The initial pair has K/N=1. These are certified finite computations, not extrapolations.

A separate exhaustive pass checks 64,276,922 ordinary prime gaps through starting prime 1,280,030,152, including the needed next endpoint, and verifies the log² bound there. Therefore success of all Sparacino points in this smaller range does not by itself identify a special advantage over all integers in the same range.

### 7.2 Structured residue-class samples

The supplied experiment examines M near 10^60 and 10^100. Four groups use twin endpoints: G prescribes 2p≡−1 modulo small primes; W greedily removes many free positions; R greedily removes few; B samples generic twins. Group C uses arbitrary integers in the W class, and D uses generic integers with the same parity and residue modulo 3. The classes are fixed before sampling.

There are 300 windows per group at 10^60 and 160 per group at 10^100: 2,760 windows in total, including 1,840 twin-based starting points. Endpoint primality uses twelve fixed strong Miller–Rabin bases; window counts use three bases after presieving through 50,000. These are probable-prime observations, not certified large-prime theorems.

The local prediction is λ_z=S_z(M)/(Θ_z log M), where S_z counts offsets avoiding primes through z. The generic Cramér prediction is N/log M. At 10^60, observed mean counts for G,W,B are 149.89,131.03,138.58, against local predictions 149.12,130.71,138.49. At 10^100 they are 253.96,217.24,230.19, against 253.57,218.61,230.60. The local predictions agree within about 1.2 reported standard errors across all twelve groups. No sampled window is empty; the maximum K/N in the four twin-based groups is about 0.06725.

These data support local-residue effects on mean counts at the sampled scales. They do not measure the extreme lower tail accurately enough to establish failures or the universal bound. A subsequent audit recomputes N, S_z, λ_z and summary statistics for all 2,760 stored windows and recounts 24 windows, including the maximum K/N in every group, with additional Miller–Rabin bases; all agree. This audit remains probabilistic at large sizes.

## 8. Reproducibility and provenance

The supplement contains the programs, numerical outputs, exact certificates, the numeric pair archive, attributed source gap table and detailed Italian research notes supporting Theorems 1–3. The manuscript is available as both PDF and editable Markdown. SHA-256 hashes are recorded in MANIFEST_SHA256.json; generated reports retain the hashes of the programs used.

Run the exact finite extension with the Python standard library; NumPy is needed for the segmented-sieve and comparison programs. The statistical experiment and its audit use the Python standard library. Commands and working directories are specified in the supplement README. Tests on illustrative tuple {0,2,6} verify CRT algebra for H=64,256,1024; this small tuple is not asserted to satisfy the Maynard prime-production step.

The historical scripts and logs remain available in the repository, but their filenames or comments suggesting a completed proof are not current mathematical claims. Version v13 is the authoritative statement of the present status. The supplied hypotheses, externally published computations and locally checked deductions are identified separately.

The author is Riccardo Sparacino. The revision was prepared with assistance from OpenAI Codex for mathematical analysis, code, checking and editing. This assistance is not an independent peer review. No priority claim is made for the derived results. The manuscript follows the Zenodo record's CC BY 4.0 licence; repository code retains its MIT licence, and external data retain their source attribution and applicable terms.

## References

[1] T. Oliveira e Silva, S. Herzog and S. Pardi. Empirical verification of the even Goldbach conjecture and computation of prime gaps up to 4·10^18. Mathematics of Computation 83 (2014), 2033–2060. Dataset and coverage: https://sweet.ua.pt/tos/gaps.html . Table: https://sweet.ua.pt/tos/gaps/t0.txt.gz . Article: https://sweet.ua.pt/tos/bib/4.12.html .

[2] J. Maynard. Dense clusters of primes in subsets. Compositio Mathematica 152 (2016), 1517–1554. Theorem 3.4, arXiv:1405.2593v2. https://arxiv.org/html/1405.2593v2 .

[3] R. C. Baker, G. Harman and J. Pintz. The difference between consecutive primes, II. Proceedings of the London Mathematical Society 83 (2001), 532–562. https://doi.org/10.1112/plms/83.3.532 .

[4] P. Erdős and M. B. Nathanson. On the sum of the reciprocals of the differences between consecutive primes. In Number Theory: New York Seminar 1991–1995 (1996), 97–101. https://www.theoryofnumbers.com/melnathanson/pdfs/nath1996-88.pdf .

[5] W. Banks, K. Ford and T. Tao. Large prime gaps and probabilistic models. Inventiones Mathematicae 233 (2023), 1471–1518. Corrected arXiv v3, 12 August 2025; especially §§1.2, 2.5, Theorem 1.6 and Lemma 6.3. https://arxiv.org/html/1908.08613v3 .

[6] H. Maier. Primes in short intervals. Michigan Mathematical Journal 32 (1985), 221–225.

[7] H. Cramér. On the order of magnitude of the difference between consecutive prime numbers. Acta Arithmetica 2 (1936), 23–46.

[8] A. Granville. Harald Cramér and the distribution of prime numbers. Scandinavian Actuarial Journal 1995, 12–28. The explicit model used here is described in [5, §1.2].
