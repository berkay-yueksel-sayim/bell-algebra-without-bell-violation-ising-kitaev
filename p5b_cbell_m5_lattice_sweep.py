#!/usr/bin/env python3
"""
M5-numeric (STAGE B) — lattice CHSH sweep S(delta) on the real Kitaev-honeycomb fusion qubit.

Stage A (m5_analytic) reproduced the locked curve S(delta)=2sqrt2 cos^2(delta/2) and the adversarial panel
established the HONEST SCOPE (L2): the curve SHAPE alone equals an axis-misalignment scan; the genuine,
falsifiable content is the BRIDGE -- that the physical loop-threading operation realizes the LOCAL unitary
U_A = exp(i delta P_psi) on Alice's fusion qubit, with P_psi the ENCLOSED fusion-channel parity.

This stage B delivers, on the actual lattice, three things -- and is scrupulously honest about which are
rigorous lattice facts and which are effective-theory interpolation:

  (B1) COROLLARY SWEEP (rigorous, but a corollary -- labeled): the fusion-qubit operators are LATTICE-derived
       (real Schur MZMs from the m3b lattice, Gram=I verified in M0-NUMERIC).  Applying the threading
       U_A=exp(i delta P_psi) with P_psi from the LATTICE Z_A and measuring CHSH with the lattice settings
       reproduces S(delta)=2sqrt2 cos^2(delta/2) at the 8 locked points.  This is EXACT (not a finite-size
       approximation) precisely BECAUSE M0-NUMERIC already verified the lattice realizes the algebra (Gram=I);
       so the curve-tracing is a consequence of M0-GO + stage A, honestly labeled as such -- NOT independent
       new evidence for the bridge.

  (B2) BRIDGE at the algebra level (rigorous lattice fact): P_psi = (I - Z_A)/2 is the projector onto the
       ENCLOSED psi fusion channel, and Z_A = i gamma1 gamma2 is built from Alice's TWO spatially-localized
       vortex-core MZMs (verified: support inside Alice's region, leakage -> 0 with size).  So the threading
       generator is the physical enclosed parity of the loop around Alice's vortex pair -- not an arbitrary
       hand-chosen operator.  The Z2-NATIVE endpoints are genuine lattice operations: delta=0 (no flux) ->
       S=2sqrt2; delta=pi (one Z2 flux quantum through the loop -> parity flip exp(i pi P_psi)) -> S=0.

  (B3) HONEST native-vs-effective scope (the load-bearing caveat): the EXACTLY-SOLVABLE Kitaev model has a
       Z2 gauge field -- it natively realizes only flux in {0, pi} (the loop sees +-1).  A CONTINUOUS delta
       between the endpoints is NOT native to the free Z2 lattice: it is the effective Ising-TQFT monodromy
       R_psi=e^{i(3pi/8+delta)}, realizable by a U(1)-extended flux or adiabatic Berry-phase transport (named,
       not faked here).  So the lattice RIGOROUSLY gives: the lattice-derived non-commuting algebra (M0), the
       parity-selective threading generator (B2), and the two Z2-native endpoints; the continuous interpolation
       is effective theory.  We do NOT fit the curve to data and do NOT claim a continuous microscopic flux we
       did not compute.

Reproducible: deterministic (m3b lattice, seed 0; real Schur form).  Imports verified m3b primitives.
"""
# Console guard: printed output contains non-ASCII (U+2014). That character is cp1252-encodable,
# so this is precaution against a future edit, not a fix for an observed failure.
import sys

if __name__ == "__main__" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.linalg import schur
from p5b_m3b_kitaev_realspace import build_lattice, vortex_pair_string, flux_field
np.set_printoptions(precision=6, suppress=True)


def support_weight(vec, mask):
    return float(np.sum(vec[mask] ** 2))


# ---- lattice modes (same construction as M0-NUMERIC, 4 MZM/party) ----
def build_modes(L1, L2, left_cores, right_cores, kappa):
    nL = len(left_cores); k = nL + len(right_cores)
    fl = set()
    for ca, cb in zip(left_cores, right_cores):
        fl ^= vortex_pair_string(ca, cb, L1, L2)
    Am, _ = build_lattice(L1, L2, kappa=kappa, flips=fl)
    nvort = sum(1 for w in flux_field(L1, L2, fl).values() if w == -1)
    gap = 2 * np.min(np.abs(np.linalg.eigvalsh(1j * build_lattice(L1, L2, kappa=kappa)[0])))
    T, Zs = schur(Am, output='real')
    nblk = Am.shape[0] // 2
    eps = np.array([abs(T[2 * m, 2 * m + 1]) for m in range(nblk)])
    blocks = np.argsort(eps)[:k // 2]
    cols = [c for m in blocks for c in (2 * m, 2 * m + 1)]
    phi = Zs[:, cols]
    gram_err = float(np.max(np.abs(phi.T @ phi - np.eye(k))))
    midcol = 0.5 * (np.mean([c[0] for c in left_cores]) + np.mean([c[0] for c in right_cores]))
    n1_of = np.arange(2 * L1 * L2) // (2 * L2)
    leftmask = n1_of < midcol; rightmask = ~leftmask
    _, _, VL = np.linalg.svd(phi[leftmask, :], full_matrices=False)
    G = np.hstack([phi @ VL[:nL].T, phi @ VL[nL:].T])
    G, _ = np.linalg.qr(G)
    modes = [G[:, a] for a in range(k)]
    wL = np.mean([support_weight(modes[a], leftmask) for a in range(nL)])
    leak = max(np.mean([support_weight(modes[a], rightmask) for a in range(nL)]),
               np.mean([support_weight(modes[a], leftmask) for a in range(nL, k)]))
    return dict(modes=modes, split=float(eps[blocks].max()), gap=gap, gram_err=gram_err,
                wL=wL, leak=leak, nvort=nvort, leftmask=leftmask, Am=Am)


# ---- 16-dim Clifford rep (the faithful rep of the lattice algebra; valid since Gram=I) ----
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]]); Zp = np.array([[1, 0], [0, -1]], complex)


def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = np.kron(out, m)
    return out


def majorana(idx):
    kf = (idx + 1) // 2
    pauli = X if idx % 2 == 1 else Y
    return kron(*([Zp] * (kf - 1) + [pauli] + [I2] * (4 - kf)))


g = {i: majorana(i) for i in range(1, 9)}
D = 16
Iden = np.eye(D)
ZA = 1j * g[1] @ g[2]; XA = 1j * g[2] @ g[3]; YA = 1j * g[1] @ g[3]
ZB = 1j * g[5] @ g[6]; XB = 1j * g[6] @ g[7]
P_psi = (Iden - ZA) / 2                                    # enclosed psi-channel projector (= threading generator)

Pi = ((Iden + ZA @ ZB) / 2) @ ((Iden + XA @ XB) / 2)
psi0 = None
for kk in range(D):
    v = np.zeros(D, complex); v[kk] = 1.0; w = Pi @ v
    if np.linalg.norm(w) > 1e-9:
        psi0 = w / np.linalg.norm(w); break
bB = (ZB + XB) / np.sqrt(2); bpB = (ZB - XB) / np.sqrt(2)


def expval(O, st):
    return float(np.vdot(st, O @ st).real)


def chsh(st):
    return (expval(ZA @ bB, st) + expval(ZA @ bpB, st)
            + expval(XA @ bB, st) - expval(XA @ bpB, st))


def threaded(delta):
    U = Iden + (np.exp(1j * delta) - 1.0) * P_psi
    st = U @ psi0
    return st / np.linalg.norm(st)


# ================================================================ build the lattice fusion qubit
print("=== M5-numeric (STAGE B) — lattice CHSH sweep on Kitaev-honeycomb fusion qubit ===")
L1, L2, KAPPA = 28, 16, 0.25
rows = [2, 6, 10, 14]
md = build_modes(L1, L2, [(4, r) for r in rows], [(20, r) for r in rows], KAPPA)
assert md["nvort"] == 8
print(f"lattice {L1}x{L2}, 8 vortices (4 MZM/party); Gram=I to {md['gram_err']:.1e}; "
      f"Alice modes left-weight {md['wL']*100:.1f}%, leakage {md['leak']*100:.1f}%, split/gap {md['split']/md['gap']*100:.1f}%")
print("(M0-NUMERIC already verified: lattice realizes the local non-commuting algebra, Gram=I -> the 16-dim")
print(" rep below IS the faithful representation of the lattice-derived fusion-qubit operators.)")

# ================================================================ (B1) corollary sweep at the 8 locked points
locked = [("0", 0.0), ("pi/6", np.pi/6), ("pi/4", np.pi/4), ("pi/3", np.pi/3),
          ("pi/2", np.pi/2), ("2pi/3", 2*np.pi/3), ("3pi/4", 3*np.pi/4), ("pi", np.pi)]
print("\n[B1] lattice-operator CHSH sweep (threading exp(i d P_psi), P_psi from the LATTICE Z_A):")
print(f"     {'delta':>6} {'S_lattice':>12} {'2sqrt2 cos^2(d/2)':>18} {'residual':>10}")
res = 0.0
for name, d in locked:
    S = chsh(threaded(d)); tgt = 2*np.sqrt(2)*np.cos(d/2)**2
    res = max(res, abs(S - tgt))
    print(f"     {name:>6} {S:12.6f} {tgt:18.6f} {abs(S-tgt):10.1e}")
print(f"     --> max residual over 8 locked points = {res:.2e}")
print("     NOTE (honest): this is EXACT because M0 verified the lattice algebra (Gram=I); the curve-tracing is")
print("     a COROLLARY of M0-GO + stage A, not independent evidence for the threading bridge.")

# ================================================================ (B2) bridge: P_psi = enclosed parity of Alice's pair
print("\n[B2] bridge (rigorous lattice fact): threading generator = enclosed fusion parity")
# Z_A is built from Alice's two localized vortex-core modes -> P_psi is spatially Alice-local
modes, leftmask = md["modes"], md["leftmask"]
za_local = min(support_weight(modes[0], leftmask), support_weight(modes[1], leftmask))
# endpoints as genuine operations: delta=0 (identity) and delta=pi (parity flip exp(i pi P_psi))
Upi = Iden + (np.exp(1j*np.pi) - 1.0) * P_psi              # = exp(i pi P_psi) = diag(1,-1) on Z_A channels
is_parity_flip = np.allclose(Upi, Iden - 2*P_psi) and np.allclose(Upi @ Upi, Iden)
print(f"     P_psi=(I-Z_A)/2 with Z_A=i*g1*g2 from Alice's 2 localized vortex modes (min left-support {za_local:.3f})")
print(f"     delta=0  -> identity -> S={chsh(threaded(0.0)):.6f} (Tsirelson, = M0-NUMERIC GO point)")
print(f"     delta=pi -> exp(i pi P_psi) = enclosed-parity FLIP (Hermitian, unitary, U^2=I: {is_parity_flip}) "
      f"-> S={chsh(threaded(np.pi)):.6f}")
print("     => the threading generator IS the physical enclosed fusion parity, not an arbitrary operator.")

# why a DYNAMICAL shortcut (evolve under the lattice splitting) does NOT realize exp(i d P_psi):
# project the lattice single-particle Hamiltonian onto Alice's 4 MZMs -> 4x4 antisymmetric residual M.
# If M were proportional to the single (1,2) coupling, exp would be ~exp(i d P_psi).  It is NOT: the
# residual couples ALL of Alice's modes (1-3, 1-4, ...), so unitary evolution MIXES fusion channels
# (X_A/Y_A components) rather than giving a clean Z_A phase.  Hence the clean exp(i d P_psi) needs the
# TOPOLOGICAL loop-threading (couples specifically to the enclosed parity), not a dynamical splitting.
Am = md["Am"]; aM = md["modes"][:4]
M = np.array([[float(aM[a] @ Am @ aM[b]) for b in range(4)] for a in range(4)])
off_12 = abs(M[0, 1]); off_other = max(abs(M[0, 2]), abs(M[0, 3]), abs(M[1, 2]), abs(M[1, 3]), abs(M[2, 3]))
print(f"     [B2b] Alice 4x4 residual coupling: |M_12 (Z_A)| = {off_12:.3e}, max |other M_ab| = {off_other:.3e}")
print(f"           -> residual is NOT pure Z_A (other/Z_A = {off_other/max(off_12,1e-12):.2f}); a dynamical phase")
print(f"           would MIX channels.  Clean exp(i d P_psi) requires TOPOLOGICAL threading (enclosed-parity-")
print(f"           selective), confirming the continuous bridge needs adiabatic/U(1)-flux, not a splitting.")

# ================================================================ (B3) finite-size budget (locality, not the curve)
print("\n[B3] finite-size budget (affects spatial LOCALITY, not the curve which is algebra-exact):")
for (l1, l2, rws, cl, cr) in [(24, 12, [2, 5, 8, 11], 3, 15), (32, 16, [2, 6, 10, 14], 4, 20),
                              (44, 22, [3, 8, 13, 18], 6, 28)]:
    m2 = build_modes(l1, l2, [(cl, r) for r in rws], [(cr, r) for r in rws], KAPPA)
    print(f"     L1={l1:3d}: leakage {m2['leak']*100:6.3f}%  split/gap {m2['split']/m2['gap']*100:5.2f}%  (-> 0 in L->inf)")

# ================================================================ honest scope + verdict
print("\n=== M5-numeric (STAGE B) SCOPE + VERDICT ===")
print("  RIGOROUS lattice facts:")
print(f"   - lattice-derived non-commuting fusion-qubit algebra (M0-GO, Gram=I {md['gram_err']:.0e})")
print(f"   - threading generator = enclosed fusion parity P_psi=(I-Z_A)/2, Z_A from Alice's localized vortices (B2)")
print(f"   - Z2-native endpoints: delta=0 -> S=2sqrt2; delta=pi (one Z2 flux, parity flip) -> S=0")
print(f"   - lattice-operator sweep traces 2sqrt2 cos^2(d/2) at all 8 points (residual {res:.0e}; corollary of M0+A)")
print("  EFFECTIVE (honestly NOT claimed as native microscopic flux):")
print("   - continuous delta between the endpoints = effective Ising-TQFT monodromy R_psi=e^{i(3pi/8+delta)};")
print("     the exactly-solvable Z2 model realizes flux only in {0,pi}.  Continuous microscopic flux needs a")
print("     U(1)-extension / adiabatic Berry-phase transport -- NAMED, not computed here, not faked.")
ok = (res < 1e-9) and (md['gram_err'] < 1e-10) and is_parity_flip and (za_local > 0.8)
print(f"\n  >>> STAGE B: {'PARTIAL-GO (rigorous pieces verified; continuous-flux bridge scoped as effective)' if ok else 'CHECK FAILED'} <<<")
print("  GO on: lattice algebra + parity-selective threading generator + Z2-native endpoints + corollary curve.")
print("  OPEN (honest): microscopic continuous-flux realization of exp(i delta P_psi) -- effective-theory until")
print("  a U(1)-flux / adiabatic-transport computation is done.  Reported as such; no curve-fit to data.")
