#!/usr/bin/env python3
"""
UNABHAENGIGE VERIFIKATION der M5-Zielkurve VOR dem Lock (Anti-Tuning).
1a-Ising-Loop-Threading-delta = PHYSIKALISCHE Relativphase auf dem max-verschraenkten Bell-Zustand
(R_psi = e^{i(3pi/8+delta)}, R NIE deformiert, YBE intakt) -- NICHT das 1b-Fibonacci-delta.
Behauptete Zielkurve (Original-Konstruktion): S(delta) = 2*sqrt(2)*cos^2(delta/2).
Eigener Code: bei FIXEN (delta=0-optimalen) Settings degradiert die fixed-setting-CHSH genau so.
"""
import numpy as np

I2=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]]); Z=np.array([[1,0],[0,-1]],complex)
kron=lambda a,b: np.kron(a,b)

def bell_state(delta):
    # |psi(delta)> = (|00> + e^{i delta}|11>)/sqrt2  -- loop-threading praegt Relativphase delta ein
    v=np.zeros(4,complex); v[0]=1/np.sqrt(2); v[3]=np.exp(1j*delta)/np.sqrt(2); return v

def exp2(A,B,st):
    O=kron(A,B); return complex(np.vdot(st,O@st)).real

def chsh_fixed(delta):
    st=bell_state(delta)
    A0,A1=Z,X                      # Alices fixe (delta=0-optimale) Settings
    B0=(Z+X)/np.sqrt(2); B1=(Z-X)/np.sqrt(2)
    return exp2(A0,B0,st)+exp2(A0,B1,st)+exp2(A1,B0,st)-exp2(A1,B1,st)

target=lambda d: 2*np.sqrt(2)*np.cos(d/2)**2

print("=== M5-Zielkurve: fixed-setting-CHSH vs 2*sqrt(2)*cos^2(delta/2) ===")
pts={'0':0,'pi/6':np.pi/6,'pi/4':np.pi/4,'pi/3':np.pi/3,'pi/2':np.pi/2,
     '2pi/3':2*np.pi/3,'3pi/4':3*np.pi/4,'pi':np.pi}
maxerr=0.0
for name,d in pts.items():
    s=chsh_fixed(d); t=target(d); maxerr=max(maxerr,abs(s-t))
    print(f"  delta={name:6s} ({d:5.3f}): CHSH={s:+.6f}  target={t:+.6f}  >2? {'JA' if s>2+1e-9 else 'nein'}")
print(f"\n  max|CHSH - 2sqrt2 cos^2(d/2)| ueber alle Punkte = {maxerr:.2e}  (0 => Kurve bestaetigt)")

# S>2-Toleranz: 2sqrt2 cos^2(d/2) = 2  ->  cos(d/2)=2^{-1/4}
dstar=2*np.arccos(2**(-0.25))
print(f"\n  S=2sqrt2 (Tsirelson) nur bei delta=0 ; S faellt monoton auf 0 bei delta=pi.")
print(f"  Nichtlokalitaets-Toleranz: S>2 fuer |delta| < delta* = {dstar:.6f} rad = {np.degrees(dstar):.2f} deg")
print(f"    (check: 2sqrt2 cos^2(dstar/2) = {target(dstar):.6f} == 2)")
# Monotonie + Deckel
fine=np.linspace(0,np.pi,2001); sf=np.array([chsh_fixed(d) for d in fine])
print(f"\n  Monoton fallend 0->pi: {bool(np.all(np.diff(sf)<=1e-9))}   max ueber [0,pi]={sf.max():.6f} (<=2sqrt2={2*np.sqrt(2):.6f})")
