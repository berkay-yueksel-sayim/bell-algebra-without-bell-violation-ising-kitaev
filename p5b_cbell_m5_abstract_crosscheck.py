#!/usr/bin/env python3
"""
INDEPENDENT VERIFICATION of the M5 target curve BEFORE the lock (anti-tuning).
1a Ising loop-threading delta = PHYSICAL relative phase on the maximally entangled Bell state
(R_psi = e^{i(3pi/8+delta)}, R NEVER deformed, YBE intact) -- NOT the 1b Fibonacci delta.
Claimed target curve (original construction): S(delta) = 2*sqrt(2)*cos^2(delta/2).
Own code: with FIXED (delta=0-optimal) settings the fixed-setting CHSH degrades in exactly the same way.
"""
# Console guard: PRECAUTION ONLY. This file contains no non-ASCII character at all, so nothing
# here can fail on a legacy console; the guard is added for uniformity across the deposit.
import sys

if __name__ == "__main__" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

I2=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]]); Z=np.array([[1,0],[0,-1]],complex)
kron=lambda a,b: np.kron(a,b)

def bell_state(delta):
    # |psi(delta)> = (|00> + e^{i delta}|11>)/sqrt2  -- loop-threading imprints the relative phase delta
    v=np.zeros(4,complex); v[0]=1/np.sqrt(2); v[3]=np.exp(1j*delta)/np.sqrt(2); return v

def exp2(A,B,st):
    O=kron(A,B); return complex(np.vdot(st,O@st)).real

def chsh_fixed(delta):
    st=bell_state(delta)
    A0,A1=Z,X                      # Alice's fixed (delta=0-optimal) settings
    B0=(Z+X)/np.sqrt(2); B1=(Z-X)/np.sqrt(2)
    return exp2(A0,B0,st)+exp2(A0,B1,st)+exp2(A1,B0,st)-exp2(A1,B1,st)

target=lambda d: 2*np.sqrt(2)*np.cos(d/2)**2

print("=== M5 target curve: fixed-setting CHSH vs 2*sqrt(2)*cos^2(delta/2) ===")
pts={'0':0,'pi/6':np.pi/6,'pi/4':np.pi/4,'pi/3':np.pi/3,'pi/2':np.pi/2,
     '2pi/3':2*np.pi/3,'3pi/4':3*np.pi/4,'pi':np.pi}
maxerr=0.0
for name,d in pts.items():
    s=chsh_fixed(d); t=target(d); maxerr=max(maxerr,abs(s-t))
    print(f"  delta={name:6s} ({d:5.3f}): CHSH={s:+.6f}  target={t:+.6f}  >2? {'YES' if s>2+1e-9 else 'no'}")
print(f"\n  max|CHSH - 2sqrt2 cos^2(d/2)| over all points = {maxerr:.2e}  (0 => curve confirmed)")

# S>2 tolerance: 2sqrt2 cos^2(d/2) = 2  ->  cos(d/2)=2^{-1/4}
dstar=2*np.arccos(2**(-0.25))
print(f"\n  S=2sqrt2 (Tsirelson) only at delta=0 ; S falls monotonically to 0 at delta=pi.")
print(f"  nonlocality tolerance: S>2 for |delta| < delta* = {dstar:.6f} rad = {np.degrees(dstar):.2f} deg")
print(f"    (check: 2sqrt2 cos^2(dstar/2) = {target(dstar):.6f} == 2)")
# monotonicity + ceiling
fine=np.linspace(0,np.pi,2001); sf=np.array([chsh_fixed(d) for d in fine])
print(f"\n  monotonically decreasing 0->pi: {bool(np.all(np.diff(sf)<=1e-9))}   max over [0,pi]={sf.max():.6f} (<=2sqrt2={2*np.sqrt(2):.6f})")
