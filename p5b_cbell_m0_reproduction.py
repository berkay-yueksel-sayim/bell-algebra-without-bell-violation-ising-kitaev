#!/usr/bin/env python3
"""
INDEPENDENT COUNTERCHECK — M0-NUMERIC GO (independent, own code, adversarial).
KEINE Dateien der Original-Konstruktion importiert. Eigene Majorana-Konstruktion (umgekehrte JW-Ordnung) +
kombinatorische Algebra-Checks (set-overlap-Regel, rep-unabhaengig) + adversariale Breaker.
Frage: ist das GO (P1=2, P2=0, P3=2sqrt2 ideal UND superselektiert, K<=2) echt + nicht getuned?
"""
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

# ---- INDEPENDENT rep: JW-String auf die ANDERE Seite (Spiegel-Ordnung), 4 Fermionen = 16-dim.
# Original-Konstruktion: g_{2k-1}=Z..Z X I.. (String links). Ich: String RECHTS, Fermion-Site gespiegelt k->5-k.
def majorana_mirror(idx):           # idx 1..8, bewusst andere Konvention
    k = (idx + 1) // 2              # fermion 1..4
    ks = 5 - k                      # gespiegelte Site
    pauli = X if idx % 2 == 1 else Y
    ops = [I2] * (ks - 1) + [pauli] + [Z] * (4 - ks)
    return kron(*ops)

g = {i: majorana_mirror(i) for i in range(1, 9)}
N = 16

# ---- (1) Clifford-Algebra in MEINER rep
herm = all(np.allclose(g[i], g[i].conj().T) for i in g)
sq = all(np.allclose(g[i] @ g[i], np.eye(N)) for i in g)
ac = all(np.allclose(g[i] @ g[j] + g[j] @ g[i], (2 * np.eye(N) if i == j else np.zeros((N, N))))
         for i in g for j in g)
print(f"[indep rep] hermitesch={herm}  quadrat=I={sq}  {{g_i,g_j}}=2d_ij={ac}")

# ---- (2) kombinatorische set-overlap-Regel (rep-UNABHAENGIG): prod_S, prod_T kommutieren
#      gdw (|S||T| - |S∩T|) gerade ist.  Gegen die Matrix-rep quergeprueft.
def comm_sign_sets(S, T):
    S, T = set(S), set(T)
    return (-1) ** ((len(S) * len(T) - len(S & T)) % 2)   # +1 = kommutiert, -1 = antikommutiert
def prod(idxs):
    M = np.eye(N, dtype=complex)
    for i in idxs:
        M = M @ g[i]
    return M
# Stichprobe: alle Paare aus relevanten Operatoren als Majorana-Sets
combo_ok = True
for S, T in itertools.combinations([(1,2),(2,3),(1,3),(5,6),(6,7),(1,2,3,4),(5,6,7,8),(1,2,3,4,5,6,7,8)], 2):
    A, B = prod(S), prod(T)
    mat_comm = np.allclose(A @ B - B @ A, 0)
    pred_comm = (comm_sign_sets(S, T) == 1)
    if mat_comm != pred_comm:
        combo_ok = False
        print(f"   MISMATCH {S} {T}: matrix_commutes={mat_comm} pred={pred_comm}")
print(f"[combinatorik] set-overlap-Regel == Matrix-rep auf allen Stichproben: {combo_ok}")

# ---- logische Operatoren (identische Definition wie gelockt)
ZA = 1j * g[1] @ g[2]; XA = 1j * g[2] @ g[3]
ZB = 1j * g[5] @ g[6]; XB = 1j * g[6] @ g[7]
PA = g[1] @ g[2] @ g[3] @ g[4]                  # Alice lokale Paritaet
PB = g[5] @ g[6] @ g[7] @ g[8]
Ptot = prod([1,2,3,4,5,6,7,8])
opn = lambda M: np.linalg.norm(M, 2)

# ---- (3) die 4 Targets, eigener Code
P1 = opn(ZA @ XA - XA @ ZA)
P2 = max(opn(a @ b - b @ a) for a in (ZA, XA) for b in (ZB, XB))
# Bell-Zustand als gemeinsamer +1-Eigenvektor von ZA ZB und XA XB
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

# ---- (4) SUPERSELEKTION (Kernfrage). psi definite globale Paritaet? beide Sektoren -> S?
par = ev(Ptot)
is_eigen = np.allclose(Ptot @ psi, par * psi, atol=1e-9)
print(f"\n[superselektion] <Ptot>={par:+.3f}  psi ist Ptot-Eigenzustand={is_eigen}")
for sgn in (+1, -1):
    proj = (np.eye(N) + sgn * Ptot) / 2
    pp = proj @ psi
    nrm = np.linalg.norm(pp)
    if nrm > 1e-9:
        pp = pp / nrm
        Ssec = chsh(ZA, XA, b, bp, st=pp)
        print(f"   Sektor P={sgn:+d} (Gewicht {nrm**2:.3f}): CHSH S = {Ssec:.6f}")
# lokale Paritaet + SSR-Vertraeglichkeit
locP = (opn(ZA@PA-PA@ZA)<1e-9 and opn(XA@PA-PA@XA)<1e-9)
ssrP = (opn(ZA@Ptot-Ptot@ZA)<1e-9 and opn(XA@Ptot-Ptot@XA)<1e-9)
locB = (opn(ZA@PB-PB@ZA)<1e-9 and opn(XA@PB-PB@XA)<1e-9)
print(f"   [ZA,PA]=[XA,PA]=0 (lokales Qubit)={locP}  settings parity-even/SSR-physikalisch={ssrP}  "
      f"settings komm. mit Bobs PB={locB}")

# ---- (5) ADVERSARIALE BREAKER
print(f"\n[adversarial]")
# (a) CHSH-Operator-Norm = Tsirelson 2sqrt2 (echtes Quanten-Max, nicht ueber-getuned)?
Cop = ZA@b + ZA@bp + XA@b - XA@bp
print(f"  (a) ||CHSH-Operator||_2 = {opn(Cop):.6f}  (Tsirelson 2sqrt2; S darf das nie ueberschreiten)")
# (b) Falsifizierbarer Gehalt: OHNE 3. Majorana hat Alice nur EIN Setting -> jedes 2. komm. mit ZA -> S<=2.
#     Scanne alle parity-even lokalen Bilineare aus NUR {g1,g2} (kein g3): bestes CHSH?
worst = 0.0
for (i,j) in [(1,2)]:
    A2 = 1j*g[i]@g[j]
    worst = max(worst, abs(chsh(ZA, A2, b, bp)))   # A2 kommutiert mit ZA -> block-diagonal
print(f"  (b) ohne 3. Majorana (2. Setting aus {{g1,g2}} -> kommutiert mit ZA): max|S|={worst:.6f}  (muss <=2)")
# (c) waere X_A ein BRAID-Unitaer statt Messung -> kein ±1-Observable
braid = np.cos(np.pi/4)*np.eye(N) + np.sin(np.pi/4)*(g[2]@g[3])
xa_is_meas = np.allclose(np.abs(np.linalg.eigvalsh(XA)), 1) and not np.allclose(XA, braid)
print(f"  (c) X_A = statisches ±1-Observable (Messung), != Braid-Unitaer exp(pi/4 g2g3): {xa_is_meas}")
# (d) psi echt verschraenkt? (reduzierte Dichte Alice-Block nicht rein)
#     Alice = Fermionen, in MEINER gespiegelten rep sitzt Alice auf qubits {3,4}; teste Schmidt ueber bipartit.
rho = np.outer(psi, psi.conj()).reshape(2,2,2,2,2,2,2,2)
rhoA = np.einsum('abcd ABcd -> abAB', rho.reshape(4,2,2,4,2,2)[:, :, :, :, :, :], optimize=True) if False else None
# robust: Schmidt-Rang via Reshape 4x4 (qubits12 | qubits34)
M = psi.reshape(4, 4)
sv = np.linalg.svd(M, compute_uv=False)
ent = -sum((s**2)*np.log2(s**2) for s in sv if s>1e-12)
print(f"  (d) Bell-Zustand Schmidt-Werte={np.round(sv,3)}  Verschraenkungs-Entropie={ent:.3f} bit (max 2)")

# ---- VERDIKT
GO = (abs(P1-2)<1e-6 and P2<1e-9 and abs(P3_ideal-2*np.sqrt(2))<1e-6 and K<=2+1e-9
      and is_eigen and locP and ssrP and abs(opn(Cop)-2*np.sqrt(2))<1e-6 and worst<=2+1e-9 and xa_is_meas)
print(f"\n=== INDEPENDENT-COUNTERCHECK VERDICT: {'GO BESTAETIGT' if GO else 'PROBLEM'} ===")
