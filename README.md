# Bell Algebra without Bell Violation: CHSH Operators for Ising Anyons in the Kitaev Honeycomb Model

**Author:** Berkay Yüksel Sayim
**ORCID:** [0009-0004-4993-7352](https://orcid.org/0009-0004-4993-7352)
**DOI (this version):** [10.5281/zenodo.21134323](https://doi.org/10.5281/zenodo.21134323)

## Abstract

We construct four spatially separated Majorana zero modes per party on a
finite Kitaev honeycomb lattice in its non-Abelian (Ising) phase and use them
to realize a local, noncommuting CHSH measurement algebra. Two static
fusion-parity observables per party commute exactly between parties and fail
to commute within a party; on the maximally entangled fusion-parity Bell
state the CHSH expectation reaches the algebraic Tsirelson value
S = 2√2 = 2.828427, surviving fermion-parity superselection exactly — a
statement specific to this four-MZM, measurement-only construction, not
to braid-then-fuse protocols. A
commuting-only control returns K = √2 ≤ 2, and the finite-size Majorana
footprint leaking across the bipartition decays monotonically with system
size. Following Howard and Vala [Phys. Rev. A 85, 022304 (2012)], we stress
that a *physical* Bell-inequality violation using only topologically
protected Ising-anyon operations does not follow from this construction —
that step requires a non-Clifford ("magic") resource, addressed in a
companion paper. A loop-threading relative phase δ reproduces the analytic
corollary S(δ) = 2√2·cos²(δ/2) to a residual of 4.4×10⁻¹⁶ on the real
lattice, and an independent gauge-invariant flux-holonomy diagnostic
confirms no adiabatic channel mixing at the quoted scale, with the sole open
discrepancy being a 0.020-rad gap between the accumulated Berry phase at one
flux quantum and the target π, traced to genuine near-degeneracies along the
loop rather than to a construction error.

## Contents

This record is compiled from `main_v1.0.tex` (RevTeX 4-2).

**Reproduction code and deposited data:**

- `p5b_cbell_m0_reproduction.py` — independent reproduction of the four locked
  M0 targets (P1, P2, S, K) using a mirrored Jordan–Wigner Majorana
  construction and a representation-independent combinatorial commutation
  rule (own code, not importing any part of the main construction). Prints
  results; values recorded in `p5b_m0_targets.json`.
- `p5b_m0_targets.json` — the four locked targets, adversarial controls, and the
  lattice-realization diagnostics (Gram error, MZM footprint leak series)
  cited in the paper.
- `p5b_cbell_m5_lattice_sweep.py` (+ dependency `p5b_m3b_kitaev_realspace.py`) — the
  actual lattice-level CHSH sweep under the loop-threading unitary, run on a
  finite Kitaev-honeycomb lattice (28×16, 8 vortices). This is the primary
  source of the residual (4.4×10⁻¹⁶) cited in the paper text.
- `p5b_cbell_m5_abstract_crosscheck.py` — an independent, self-contained
  (lattice-free) fixed-setting reproduction of the same curve, used only as
  a second, unrelated cross-check (its own residual, 8.88×10⁻¹⁶, is recorded
  in `p5b_m5_curve.json` but is *not* the value cited in the paper text).
- `p5b_m5_curve.json` — both routes above, clearly labeled, plus the
  loop-threading bridge diagnostics and the nonlocality threshold δ*.
- `p5b_cbell_m5_flux_holonomy.py` + `p5b_m5_flux.json` — the gauge-invariant
  continuous-flux Wilson-holonomy diagnostic (Sec. "Continuous-flux
  holonomy: an honest partial"): near-trivial holonomy
  (‖[W,Z_A]‖_F = 0.0132), with the single open discrepancy (criterion c3,
  δ_full mod 2π = 3.1212 vs. target π) traced to genuine near-degeneracies
  along the flux loop.

All scripts require NumPy (and SciPy for `p5b_cbell_m5_lattice_sweep.py`) and are
deterministic.

## License
- Paper, figures, and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `LICENSE`
- Source code (`*.py`): [MIT License](https://opensource.org/licenses/MIT) — see `LICENSE-CODE`
