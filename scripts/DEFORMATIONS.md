# Deformation matrices and stress-to-Cij mapping (frozen convention)

Reference cell: cubic, lattice constant `a`. Applied deformation acts on
lattice vectors as `A' = (I + H) A`. Infinitesimal strain `e = (H+H^T)/2`.
QE prints stress `S` with `P_printed = +Tr(S)/3` (verified: compression gives
positive values). True Cauchy stress `C = -S`. Elastic constants are
second derivatives of energy, extracted as follows (signs calibrated by
matching stress-derived C11 against energy-curvature C11 on NiOs: agree
266.7 vs 269.2 GPa).

## Normal strains (C11, C12)

- `xx` amplitude d: `H_xx = d` (a-vector scaled by 1+d).
  `e_xx = d`. `C11 = -m_xx/10`, `C12 = -m_trans/10`, where `m` are slopes of
  printed stress (kbar) vs d, `/10` converts kbar to GPa, minus sign undoes
  QE's flipped sign convention.
- Same for `yy`, `zz`. The three axes must agree (cubic check).
- Energy cross-check (sign-free): `C11 = (1/Omega) d2E/dd2`,
  Omega in bohr^3, E in Ry, `1 Ry/bohr^3 = 14710.5 GPa`.

## Shear strains (C44)

- `xy` amplitude d: `a_vec[i] += (d/2) b_vec[i]`, i.e. `H_xy = d/2`,
  `e_xy = d/4`. True `s12 = C44*d/2`; with QE's flipped sign the measured
  slope is `-C44/2`, so `C44[GPa] = -2*slope_kbar/10`. Verified: NiOs gives
  229.0 vs energy-curvature 227.9; RuO gives -147.7 vs -148.1.
- Same for `xz`, `yz`.

## Volume-conserving tetragonal scan (C11-C12 sign check, no conventions)

- `a' = a(1+d)`, `b' = a(1+d)`, `c' = a/(1+d)^2`. Volume exact to all orders.
- `dE = 3 Omega (C11-C12) d^2 + O(d^3)`.
- Downward-opening parabola (dip at d != 0) PROVES `C11-C12 < 0` with no
  stress-sign, unit, or axis-label ambiguity.

## EOS (B0 check)

- Isotropic scaling `s` in {0.97..1.03}, fractional positions frozen.
- Birch-Murnaghan fit gives B0, V0. Must satisfy `B0 ~= (C11+2C12)/3`.
