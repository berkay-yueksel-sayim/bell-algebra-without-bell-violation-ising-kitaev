#!/usr/bin/env python3
"""
INDEPENDENT COUNTERCHECK — M0-NUMERIC GO (independent, own code, adversarial).
NO file of the original construction is imported. Own Majorana construction (reversed JW ordering) +
combinatorial algebra checks (set-overlap rule, rep-independent) + adversarial breakers.
Question: is the GO (P1=2, P2=0, P3=2sqrt2 ideal AND superselected, K<=2) genuine and not tuned?
"""
# Console guard: printed output contains non-ASCII (U+00B1). That character is cp1252-encodable,
# so this is precaution against a future edit, not a fix for an observed failure.
import sys

if __name__ == "__main__" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import itertools

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)

def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = np.kron(out, m)
    return out

# ---- INDEPENDENT rep: JW string on the OTHER side (mirrored ordering), 4 fermions = 16-dim.
# Original construction: g_{2k-1}=Z..Z X I.. (string on the left). Mine: string on the RIGHT, fermion site mirrored k->5-k.
def majorana_mirror(idx):           # idx 1..8, deliberately a different convention
    k = (idx + 1) // 2              # fermion 1..4
    ks = 5 - k                      # mirrored site
    pauli = X if idx % 2 == 1 else Y
    ops = [I2] * (ks - 1) + [pauli] + [Z] * (4 - ks)
    return kron(*ops)

g = {i: majorana_mirror(i) for i in range(1, 9)}
N = 16

# ---- (1) Clifford algebra in MY OWN rep
herm = all(np.allclose(g[i], g[i].conj().T) for i in g)
sq = all(np.allclose(g[i] @ g[i], np.eye(N)) for i in g)
ac = all(np.allclose(g[i] @ g[j] + g[j] @ g[i], (2 * np.eye(N) if i == j else np.zeros((N, N))))
         for i in g for j in g)
print(f"[indep rep] hermitian={herm}  square=I={sq}  {{g_i,g_j}}=2d_ij={ac}")

# ---- (2) combinatorial set-overlap rule (rep-INDEPENDENT): prod_S, prod_T commute
#      iff (|S||T| - |S∩T|) is even.  Cross-checked against the matrix rep.
def comm_sign_sets(S, T):
    S, T = set(S), set(T)
    return (-1) ** ((len(S) * len(T) - len(S & T)) % 2)   # +1 = commutes, -1 = anticommutes
def prod(idxs):
    M = np.eye(N, dtype=complex)
    for i in idxs:
        M = M @ g[i]
    return M
# Sample: all pairs of the relevant operators as Majorana sets
combo_ok = True
for S, T in itertools.combinations([(1,2),(2,3),(1,3),(5,6),(6,7),(1,2,3,4),(5,6,7,8),(1,2,3,4,5,6,7,8)], 2):
    A, B = prod(S), prod(T)
    mat_comm = np.allclose(A @ B - B @ A, 0)
    pred_comm = (comm_sign_sets(S, T) == 1)
    if mat_comm != pred_comm:
        combo_ok = False
        print(f"   MISMATCH {S} {T}: matrix_commutes={mat_comm} pred={pred_comm}")
print(f"[combinatorics] set-overlap rule == matrix rep on all samples: {combo_ok}")

# ---- logical operators (definition identical to the locked one)
ZA = 1j * g[1] @ g[2]; XA = 1j * g[2] @ g[3]
ZB = 1j * g[5] @ g[6]; XB = 1j * g[6] @ g[7]
PA = g[1] @ g[2] @ g[3] @ g[4]                  # Alice's local parity
PB = g[5] @ g[6] @ g[7] @ g[8]
Ptot = prod([1,2,3,4,5,6,7,8])
opn = lambda M: np.linalg.norm(M, 2)

# ---- (3) the 4 targets, own code
P1 = opn(ZA @ XA - XA @ ZA)
P2 = max(opn(a @ b - b @ a) for a in (ZA, XA) for b in (ZB, XB))
# Bell state as the joint +1 eigenvector of ZA ZB and XA XB
Pi = ((np.eye(N) + ZA @ ZB) / 2) @ ((np.eye(N) + XA @ XB) / 2)
psi = None
for k in range(N):
    v = np.zeros(N, complex); v[k] = 1.0; w = Pi @ v
    if np.linalg.norm(w) > 1e-9:
        psi = w / np.linalg.norm(w); break
ev = lambda O: complex(np.vdot(psi, O @ psi)).real
def chsh(a, ap, b, bp, st=None):
    e = (lambda O: complex(np.vdot(st, O @ st)).real) if st is not None else ev
    return e(a @ b) + e(a @ bp) + e(ap @ b) - e(ap @ bp)
b = (ZB + XB) / np.sqrt(2); bp = (ZB - XB) / np.sqrt(2)
P3_ideal = chsh(ZA, XA, b, bp)
K = chsh(ZA, ZA, b, bp)
print(f"\n[targets] P1=||[ZA,XA]||={P1:.6f} (2)  P2=max||[A,B]||={P2:.2e} (0)  "
      f"P3_ideal={P3_ideal:.6f} (2sqrt2={2*np.sqrt(2):.6f})  K_commuting={K:.6f} (<=2)")

# ---- (4) SUPERSELECTION (core question). Does psi have definite global parity? both sectors -> S?
par = ev(Ptot)
is_eigen = np.allclose(Ptot @ psi, par * psi, atol=1e-9)
print(f"\n[superselection] <Ptot>={par:+.3f}  psi is Ptot eigenstate={is_eigen}")
for sgn in (+1, -1):
    proj = (np.eye(N) + sgn * Ptot) / 2
    pp = proj @ psi
    nrm = np.linalg.norm(pp)
    if nrm > 1e-9:
        pp = pp / nrm
        Ssec = chsh(ZA, XA, b, bp, st=pp)
        print(f"   sector P={sgn:+d} (weight {nrm**2:.3f}): CHSH S = {Ssec:.6f}")
# local parity + SSR compatibility
locP = (opn(ZA@PA-PA@ZA)<1e-9 and opn(XA@PA-PA@XA)<1e-9)
ssrP = (opn(ZA@Ptot-Ptot@ZA)<1e-9 and opn(XA@Ptot-Ptot@XA)<1e-9)
locB = (opn(ZA@PB-PB@ZA)<1e-9 and opn(XA@PB-PB@XA)<1e-9)
print(f"   [ZA,PA]=[XA,PA]=0 (local qubit)={locP}  settings parity-even/SSR-physical={ssrP}  "
      f"settings commute with Bob's PB={locB}")

# ---- (5) ADVERSARIAL BREAKERS
print(f"\n[adversarial]")
# (a) CHSH operator norm = Tsirelson 2sqrt2 (true quantum maximum, not over-tuned)?
Cop = ZA@b + ZA@bp + XA@b - XA@bp
print(f"  (a) ||CHSH operator||_2 = {opn(Cop):.6f}  (Tsirelson 2sqrt2; S may never exceed it)")
# (b) Falsifiable content: WITHOUT a 3rd Majorana Alice has only ONE setting -> any 2nd commutes with ZA -> S<=2.
#     Scan all parity-even local bilinears built from ONLY {g1,g2} (no g3): best CHSH?
worst = 0.0
for (i,j) in [(1,2)]:
    A2 = 1j*g[i]@g[j]
    worst = max(worst, abs(chsh(ZA, A2, b, bp)))   # A2 commutes with ZA -> block-diagonal
print(f"  (b) without a 3rd Majorana (2nd setting from {{g1,g2}} -> commutes with ZA): max|S|={worst:.6f}  (must be <=2)")
# (c) were X_A a BRAID unitary instead of a measurement -> no ±1 observable
braid = np.cos(np.pi/4)*np.eye(N) + np.sin(np.pi/4)*(g[2]@g[3])
xa_is_meas = np.allclose(np.abs(np.linalg.eigvalsh(XA)), 1) and not np.allclose(XA, braid)
print(f"  (c) X_A = static ±1 observable (measurement), != braid unitary exp(pi/4 g2g3): {xa_is_meas}")
# (d) is psi genuinely entangled? (reduced density of Alice's block not pure)
#     Alice = fermions; in MY mirrored rep Alice sits on qubits {3,4}; test Schmidt across the bipartition.
rho = np.outer(psi, psi.conj()).reshape(2,2,2,2,2,2,2,2)
rhoA = np.einsum('abcd ABcd -> abAB', rho.reshape(4,2,2,4,2,2)[:, :, :, :, :, :], optimize=True) if False else None
# robust: Schmidt rank via reshape 4x4 (qubits12 | qubits34)
M = psi.reshape(4, 4)
sv = np.linalg.svd(M, compute_uv=False)
ent = -sum((s**2)*np.log2(s**2) for s in sv if s>1e-12)
print(f"  (d) Bell state Schmidt values={np.round(sv,3)}  entanglement entropy={ent:.3f} bit (max 2)")

# ---- VERDICT
GO = (abs(P1-2)<1e-6 and P2<1e-9 and abs(P3_ideal-2*np.sqrt(2))<1e-6 and K<=2+1e-9
      and is_eigen and locP and ssrP and abs(opn(Cop)-2*np.sqrt(2))<1e-6 and worst<=2+1e-9 and xa_is_meas)
print(f"\n=== INDEPENDENT-COUNTERCHECK VERDICT: {'GO CONFIRMED' if GO else 'PROBLEM'} ===")
