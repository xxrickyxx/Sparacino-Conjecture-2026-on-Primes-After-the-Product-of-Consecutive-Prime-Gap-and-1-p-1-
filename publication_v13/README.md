# Supplement to version 13

This folder contains the version 13 research note in PDF and editable Markdown, a reproducibility archive, and a SHA-256 manifest. The main manuscript is the authoritative statement of the results and their limits.

The finite theorem uses the published exhaustive prime-gap table as an external input; the original scan to (4\times10^{18}) was not rerun. The accompanying certificate checks the table's counts, boundary prime, exact logarithmic inequalities, and transfer to the conjecture. The separate Maynard argument concerns actual primes and is stated with ineffective constants. The random-model theorem concerns only the explicitly defined random set. Large-scale residue experiments use probable-prime tests, not certified primality tests.

The archive also includes the older GPU outputs needed to audit withdrawn claims. The 128-bit program's first 15 Miller–Rabin bases are not deterministic over its claimed range; version 13 documents a composite number that passes all 15. The illustrative CRT tests verify congruence algebra for the small tuple \(\{0,2,6\}\); they do not verify the Maynard prime-production step.

## Reproducing the checks

Run the commands from the archive's top-level directory with Python 3.12.14. The exact finite-gap certificate uses only the standard library:

```text
python certifica_estensione_finita.py
```

The audit of the original GPU artifact and the independent scan uses NumPy:

```text
python audit_riproducibile.py
python verifica_indipendente.py --limit 10000000 --count-to 2900000000
```

The second command repeats a segmented exact sieve through 2.9 billion and the exact check through \(p_1\le10^7\); it is computationally heavier than the certificate checks.

The residue data and small CRT verifications can be checked with:

```text
python verifica_avanzamento_rigoroso.py
python verifica_costruzione_crt.py
python verifica_teorema_condizionale.py
```

The finite experiment and threshold outputs are included. Re-running their producers is optional and takes longer:

```text
python esperimento_classi_residue.py
python soglia_modello_raffinato.py
```

`verifica_avanzamento_rigoroso.py` checks all 2,760 stored windows and recomputes the 24 selected windows with extra Miller–Rabin bases. Those checks remain probabilistic for the large endpoints and candidates.

## Included inputs and licenses

`fonti_esterne/oliveira_prime_gaps_t0.txt.gz` is the attributed gap table from Oliveira e Silva, Herzog and Pardi, available at https://sweet.ua.pt/tos/gaps/t0.txt.gz. It is included as the stated external input. `fonti_originali/` contains selected historical scripts and outputs; its included `LICENSE` is MIT. The Zenodo record retains its CC BY 4.0 license for the version 13 note and accompanying material. The source table remains attributed to its authors and source.

The package manifest lists SHA-256 hashes for every included file other than the manifest itself. The JSON outputs produced by the scripts also record the hashes of the scripts used to generate them.
