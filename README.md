# Sparacino Conjecture (2026)

This repository collects historical code and materials for the Sparacino conjecture on primes after the product of a consecutive prime gap and its lower prime. **The universal conjecture remains open.** The current research note is version 13 (25 October 2026), published on [Zenodo](https://doi.org/10.5281/zenodo.23638603). The [concept DOI](https://doi.org/10.5281/zenodo.19076295) resolves to the latest version.

## Current statement and results

For consecutive primes \(p<p+d\), put \(M=dp\) and \(N=\lceil(\log M)^2\rceil\). The conjecture says that \((M,M+N]\) always contains a prime. Version 13 does not claim a proof or a counterexample.

The note verifies the assertion for the finite range \(M<3,999,999,999,999,999,887\), conditional on the correctness of the published exhaustive prime-gap table of Oliveira e Silva, Herzog and Pardi. That external scan was not rerun; the reduction from its data and the logarithmic inequalities are checked in the accompanying certificate.

It also gives an unconditional deduction from Maynard's theorem: infinitely many consecutive prime pairs have one fixed bounded gap and first-prime distance \(K\ge a\log\log M\), for an absolute constant \(a>0\). This does not establish the conjectured \(K\le\lceil(\log M)^2\rceil\) bound. A separate theorem concerns only the explicitly defined Granville random model. Probable-prime experiments are identified as probabilistic, not as certified primality proofs.

Earlier repository descriptions and historical files are retained for provenance. Where they conflict with the corrected scope in version 13, use the version 13 note as the current statement.

## Version 13 files and reproducibility

The publication files, SHA-256 manifest and complete reproducibility archive are in [`publication_v13/`](publication_v13/). The archive includes the scripts, inputs, data, and checks needed to reproduce the stated audits. Begin with [`publication_v13/README.md`](publication_v13/README.md) for commands and computational requirements. The PDF and editable Markdown note are also available directly from Zenodo.

The code in this repository is licensed under MIT as stated in [`LICENSE`](LICENSE). The version 13 publication is distributed under the license indicated by its Zenodo record (CC BY 4.0); the included external prime-gap data remains attributed to its authors and source.
