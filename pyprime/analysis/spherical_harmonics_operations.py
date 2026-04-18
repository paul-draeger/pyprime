import pandas as pd
import numpy as np
#if not hasattr(np, 'bool'):
#    np.bool = np.bool_
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from .analyse import analyse_Re
from matplotlib import rc
import os
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

from scipy.special import sph_harm
from scipy.spatial import SphericalVoronoi, geometric_slerp
import pickle


def compute_spherical_harmonics(NF, TT, PP, ni, mi):
    """
    Compute spherical harmonics and derivatives.
    """
    sinT = np.sin(TT)
    cosT = np.cos(TT)

    NZ  = round((NF+1)*(NF+2)/2)

    Pnm = np.zeros([NZ,np.shape(TT)[0],np.shape(TT)[1]])
    PnmdT = np.zeros([NZ,np.shape(TT)[0],np.shape(TT)[1]])

    Pnm[0,:,:] = 1.0               # = Pnm(zz(0,0),:) = P_00
    Pnm[1,:,:] = cosT               # = Pnm(zz(1,0),:) = P_10
    Pnm[2,:,:] = sinT               # = Pnm(zz(1,2),:) = P_11
    Pnm[3,:,:] = 0.5*(3*cosT**2-1)# = Pnm(zz(2,0),:) = P_20
    Pnm[4,:,:] = 3*cosT*sinT        # = Pnm(zz(2,1),:) = P_21
    Pnm[5,:,:] = 3*sinT**2          # = Pnm(zz(2,2),:) = P_22

    # Recursive computation
    for n in range(3, NF + 1):
        for m in range(0, n - 1):
            Pnm[zz(n, m), :, :] = (1.0 / (n - m)) * ((2 * n - 1) * cosT * Pnm[zz(n - 1, m), :, :] - (n + m - 1) * Pnm[zz(n - 2, m), :, :])

        m = n - 1
        Pnm[zz(n, m), :, :] = (2 * n - 1) * cosT * Pnm[zz(n, -1), :, :]

        m = n
        Pnm[zz(n, m), :, :] = (2 * n - 1) * sinT * Pnm[zz(n, -1), :, :]

    # Derivatives of P_nm
    for n in range(1, NF + 1):
        m = 0
        PnmdT[zz(n, m), :, :] = -Pnm[zz(n, 1), :, :]

        for m in range(1, n):
            PnmdT[zz(n, m), :, :] = 0.5 * ((n + m) * (n - m + 1) * Pnm[zz(n, m - 1), :, :] - Pnm[zz(n, m + 1), :, :])

        m = n
        PnmdT[zz(n, m), :, :] = n * Pnm[zz(n, n - 1), :, :]

    #eimP(0:NF,sh%nl)
    # Build spherical harmonics and derivatives
    eimP = np.zeros([NF + 1, np.shape(sinT)[0], np.shape(sinT)[1]], dtype=complex)
    for m in range(0, NF + 1):
        eimP[m, :, :] = np.exp(1j * m * PP )

    n = ni
    m = mi
    Y = np.zeros_like(Pnm, dtype=complex)
    Ydp = np.zeros_like(Pnm, dtype=complex)
    Ydt = np.zeros_like(Pnm, dtype=complex)
    for n in range(0,NF+1):
        for m in range(0,n+1):
            Z = zz(n, m)
            Y[Z,:,:]   = fnm(n,m) * eimP[m, :, :] * Pnm[Z, :, :]
            Ydp[Z,:,:] = 1j * m * Y[Z,:,:]
            Ydt[Z,:,:] = fnm(n,m) * eimP[m, :, :] * PnmdT[Z, :, :]

    return Y, Ydp, Ydt


def zz(n, m):
    """
    Vectorized z-index for triangle-shaped n-m-space.
    """
    zz = round(n*(n+1)/2 + m )
    #print(zz)
    return zz

from math import sqrt, factorial, pi

def fnm(n,m):
    """
    Create and compute the fnm array based on the given parameters.

    Parameters:
    - NZ: Size of the fnm array (must be pre-defined)
    - NF: The maximum degree of the factorial calculations

    Returns:
    - fnm: A NumPy array of calculated values
    """

    fnm = np.sqrt(( (2*n+1)*factorial(n-m) )/( 4*np.pi*factorial(n+m) ))

    return fnm


def compute_spherical_harmonics_fibo(NF, TT, PP):
    """
    Compute spherical harmonics and derivatives.
    """

    sinT = np.sin(TT)
    cosT = np.cos(TT)

    NZ  = round((NF+1)*(NF+2)/2)

    Pnm = np.zeros([NZ,np.size(TT)])
    PnmdT = np.zeros([NZ,np.size(TT)])

    Pnm[0,:] = 1.0               # = Pnm(zz(0,0),:) = P_00
    Pnm[1,:] = cosT               # = Pnm(zz(1,0),:) = P_10
    Pnm[2,:] = sinT               # = Pnm(zz(1,2),:) = P_11
    Pnm[3,:] = 0.5*(3*cosT**2-1)# = Pnm(zz(2,0),:) = P_20
    Pnm[4,:] = 3*cosT*sinT        # = Pnm(zz(2,1),:) = P_21
    Pnm[5,:] = 3*sinT**2          # = Pnm(zz(2,2),:) = P_22

    # Recursive computation
    for n in range(3, NF + 1):
        for m in range(0, n - 1):
            Pnm[zz(n, m), :] = (1.0 / (n - m)) * ((2 * n - 1) * cosT * Pnm[zz(n - 1, m), :] - (n + m - 1) * Pnm[zz(n - 2, m), :])

        m = n - 1
        Pnm[zz(n, m), :] = (2 * n - 1) * cosT * Pnm[zz(n, -1), :]

        m = n
        Pnm[zz(n, m), :] = (2 * n - 1) * sinT * Pnm[zz(n, -1), :]

    # Derivatives of P_nm
    for n in range(1, NF + 1):
        m = 0
        PnmdT[zz(n, m), :] = -Pnm[zz(n, 1), :]

        for m in range(1, n):
            PnmdT[zz(n, m), :] = 0.5 * ((n + m) * (n - m + 1) * Pnm[zz(n, m - 1), :] - Pnm[zz(n, m + 1), :])

        m = n
        PnmdT[zz(n, m), :] = n * Pnm[zz(n, n - 1), :]

    #eimP(0:NF,sh%nl)
    # Build spherical harmonics and derivatives
    eimP = np.zeros([NF + 1, np.size(sinT)], dtype=complex)
    for m in range(0, NF + 1):
        eimP[m, :] = np.exp(1j * m * PP )

    Y = np.zeros_like(Pnm, dtype=complex)
    Ydp = np.zeros_like(Pnm, dtype=complex)
    Ydt = np.zeros_like(Pnm, dtype=complex)
    for n in range(0,NF+1):
        for m in range(0,n+1):
            Z = zz(n, m)
            Y[Z,:]   = fnm(n,m) * eimP[m, :] * Pnm[Z, :]

            Ydp[Z,:] = 1j * m * Y[Z,:]
            Ydt[Z,:] = fnm(n,m) * eimP[m, :] * PnmdT[Z, :]

    return Y, Ydp, Ydt



def compute_spherical_harmonics_2d(NF, TT, PP, ni, mi):
    """
    Compute spherical harmonics and derivatives.
    """

    sinT = np.sin(TT)
    cosT = np.cos(TT)

    NZ  = round((NF+1)*(NF+2)/2)

    Pnm    = np.zeros([NZ,np.shape(TT)[0],np.shape(TT)[1]])
    PnmdT  = np.zeros([NZ,np.shape(TT)[0],np.shape(TT)[1]])
    PnmdTT = np.zeros([NZ,np.shape(TT)[0],np.shape(TT)[1]])

    Pnm[0,:,:] = 1.0               # = Pnm(zz(0,0),:) = P_00
    Pnm[1,:,:] = cosT               # = Pnm(zz(1,0),:) = P_10
    Pnm[2,:,:] = sinT               # = Pnm(zz(1,2),:) = P_11
    Pnm[3,:,:] = 0.5*(3*cosT**2-1)# = Pnm(zz(2,0),:) = P_20
    Pnm[4,:,:] = 3*cosT*sinT        # = Pnm(zz(2,1),:) = P_21
    Pnm[5,:,:] = 3*sinT**2          # = Pnm(zz(2,2),:) = P_22

    # Recursive computation
    for n in range(3, NF + 1):
        for m in range(0, n - 1):
            Pnm[zz(n, m), :, :] = (1.0 / (n - m)) * ((2 * n - 1) * cosT * Pnm[zz(n - 1, m), :, :] - (n + m - 1) * Pnm[zz(n - 2, m), :, :])

        m = n - 1
        Pnm[zz(n, m), :, :] = (2 * n - 1) * cosT * Pnm[zz(n, -1), :, :]

        m = n
        Pnm[zz(n, m), :, :] = (2 * n - 1) * sinT * Pnm[zz(n, -1), :, :]

    # Derivatives of P_nm
    for n in range(1, NF + 1):
        m = 0
        PnmdT[zz(n, m), :, :] = -Pnm[zz(n, 1), :, :]

        for m in range(1, n):
            PnmdT[zz(n, m), :, :] = 0.5 * ((n + m) * (n - m + 1) * Pnm[zz(n, m - 1), :, :] - Pnm[zz(n, m + 1), :, :])

        m = n
        PnmdT[zz(n, m), :, :] = n * Pnm[zz(n, n - 1), :, :]

    # second derivatives of P_nm (Bosch2000)
    for n in range(1, NF + 1):
        m = 0
        PnmdTT[zz(n, m), :, :] = -PnmdT[zz(n, 1), :, :]

        for m in range(1, n):
            PnmdTT[zz(n, m), :, :] = 0.5 * ((n + m) * (n - m + 1) * PnmdT[zz(n, m - 1), :, :] - PnmdT[zz(n, m + 1), :, :])

        m = n
        PnmdTT[zz(n, m), :, :] = n * PnmdT[zz(n, n - 1), :, :]

    #eimP(0:NF,sh%nl)
    # Build spherical harmonics and derivatives
    eimP = np.zeros([NF + 1, np.shape(sinT)[0], np.shape(sinT)[1]], dtype=complex)
    for m in range(0, NF + 1):
        eimP[m, :, :] = np.exp(1j * m * PP )

    n = ni
    m = mi
    Y = np.zeros_like(Pnm, dtype=complex)
    Ydp = np.zeros_like(Pnm, dtype=complex)
    Ydt = np.zeros_like(Pnm, dtype=complex)
    Ydpp = np.zeros_like(Pnm, dtype=complex)
    Ydtt = np.zeros_like(Pnm, dtype=complex)
    Ydtp = np.zeros_like(Pnm, dtype=complex)
    for n in range(0,NF+1):
        for m in range(0,n+1):
            Z = zz(n, m)
            Y[Z,:,:]   = fnm(n,m) * eimP[m, :, :] * Pnm[Z, :, :]
            Ydp[Z,:,:] = 1j * m * Y[Z,:,:]
            Ydt[Z,:,:] = fnm(n,m) * eimP[m, :, :] * PnmdT[Z, :, :]
            Ydpp[Z,:,:] =   -m**2 * Y[Z,:,:]
            Ydtt[Z,:,:] =  fnm(n,m) * eimP[m, :, :] * PnmdTT[Z, :, :]
            Ydtp[Z,:,:] =  1j * m * Ydt[Z,:,:]

    return Y, Ydp, Ydt, Ydtt, Ydpp, Ydtp



def read_tdesign_nodes(filename):
    """
    Reads a t-design node file and returns an array X of shape (3, N).
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # find number of nodes
    n_nodes = None
    for i, line in enumerate(lines):
        if "** Nodes" in line:
            n_nodes = int(lines[i + 1].split()[0])
            break
    if n_nodes is None:
        raise ValueError("Number of nodes not found.")

    # find start of coordinates
    start = None
    for i, line in enumerate(lines):
        if "** Nodal coordinates" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError("Nodal coordinates section not found.")

    # read coordinates
    coords = []
    for line in lines[start:start + n_nodes]:
        x, y, z = map(float, line.split())
        coords.append([x, y, z])

    X = np.array(coords).T  # shape (3, N)
    return X


def sh_calc_weights(X, Y, NF, w0=None, prec=1e-12, maxiter=None, check_positive=True):
    """
    Compute quadrature weights for an arbitrary quadrature mesh, analogous to the
    Fortran subroutine sh_calcWeights.

    Parameters
    ----------
    X : (3, NL) or (NL, 3) array_like
        Quadrature points (only used for NL consistency checks).
    Y : (Z, NL) complex ndarray
        Spherical harmonics evaluated at the NL points, packed by
        z = round(n*(n+1)/2 + m) for m=0..n, and using the same packing for
        the positive-m entries needed to construct the sine (imag) block.
        Must contain all indices up to z_max = NF*(NF+1)//2 + NF.
    NF : int
        Desired spherical harmonics order.
    w0 : (NL,) array_like, optional
        Initial weights (used as initial guess for CG). If None, starts from zeros.
    prec : float, optional
        Relative residual tolerance for conjugate gradient on normal equations.
    maxiter : int, optional
        Max CG iterations. Default: 10*NL.
    check_positive : bool, optional
        If True, raise ValueError if any weight is negative.

    Returns
    -------
    w : (NL,) ndarray
        Quadrature weights.

    See also
    --------
    The constructed system is A w ≈ b with:
      b[0] = sqrt(4*pi), all others 0,
      A stacks the "cos" (real) and "sin" (-(-1)^m imag) parts in the same way
      as the Fortran code.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim != 2 or not np.iscomplexobj(Y):
        raise ValueError("Y must be a 2D complex array of shape (Z, NL).")

    NL = Y.shape[1]
    if X.shape[0] == 3 and X.shape[1] == NL:
        pass
    elif X.shape[1] == 3 and X.shape[0] == NL:
        pass
    else:
        raise ValueError(f"X must have shape (3, NL) or (NL, 3) with NL={NL}.")

    # sizes exactly as in the Fortran code
    nA1 = (NF + 1) * (NF + 2) // 2          # m = 0..n
    nA2 = NF * (NF + 1) // 2                # m = -n..-1
    nrow = (NF + 1) ** 2

    z_max = NF * (NF + 1) // 2 + NF
    if Y.shape[0] <= z_max:
        raise ValueError(
            f"Y has too few rows (Z={Y.shape[0]}). Need at least {z_max+1}."
        )

    A1 = np.empty((NL, nA1), dtype=float)
    A2 = np.empty((NL, nA2), dtype=float)

    # Build A1 and A2 using the same indexing pattern as the Fortran routine.
    # Use k0 = n(n+1)/2 (0-based version of k = n(n+1)/2 + 1).
    for n in range(NF + 1):
        k0 = n * (n + 1) // 2

        # m = 0..n -> A1[:, k0+m] = Re(Y[z, :]) with z = k0+m
        for m in range(0, n + 1):
            z = int(round(k0 + m))
            A1[:, k0 + m] = np.real(Y[z, :])

        # m = -n..-1 only for n>=1 -> A2[:, k0+m] = -(-1)^m * Im(Y[k0-m, :])
        # Note: k0-m = k0 + |m| points to the corresponding positive-m entry.
        if n >= 1:
            for m in range(-n, 0):
                col = k0 + m  # 0..nA2-1
                zpos = int(round(k0 - m))  # k0 + |m|
                A2[:, col] = -(((-1.0) ** m) * np.imag(Y[zpos, :]))

    # Stack into A of shape (nrow, NL): first transpose(A1), then transpose(A2)
    A = np.empty((nrow, NL), dtype=float)
    A[:nA1, :] = A1.T
    A[nA1:, :] = A2.T

    # b vector
    b = np.zeros(nrow, dtype=float)
    b[0] = np.sqrt(4.0 * np.pi)

    # Solve least squares min ||A w - b|| via CG on normal equations:
    # (A^T A) w = A^T b
    def AtA_dot(v):
        Av = A @ v
        return A.T @ Av

    rhs = A.T @ b

    if w0 is None:
        w = np.zeros(NL, dtype=float)
    else:
        w = np.asarray(w0, dtype=float).reshape(-1)
        if w.shape[0] != NL:
            raise ValueError(f"w0 must have length NL={NL}.")

    if maxiter is None:
        maxiter = 10 * NL

    # Conjugate Gradient (SPD assumed; AtA is PSD; should be OK in typical quadrature setups)
    r = rhs - AtA_dot(w)
    p = r.copy()
    rr_old = float(r @ r)

    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm == 0.0:
        rhs_norm = 1.0

    for _ in range(maxiter):
        Ap = AtA_dot(p)
        denom = float(p @ Ap)
        if denom <= 0.0:
            # Degeneracy or numerical issues; fall back to least squares
            w = np.linalg.lstsq(A, b, rcond=None)[0]
            break

        alpha = rr_old / denom
        w += alpha * p
        r -= alpha * Ap

        rel_res = np.linalg.norm(r) / rhs_norm
        if rel_res <= prec:
            break

        rr_new = float(r @ r)
        beta = rr_new / rr_old
        p = r + beta * p
        rr_old = rr_new
    else:
        # If CG didn't converge, do a safe fallback
        w = np.linalg.lstsq(A, b, rcond=None)[0]

    if check_positive:
        nneg = int(np.sum(w < 0.0))
        if nneg > 0:
            raise ValueError(
                f"Negative quadrature weights detected: {nneg}. "
                "The quadrature may be ill-conditioned or insufficiently sampled."
            )

    return w