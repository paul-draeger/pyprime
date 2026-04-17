import numpy as np
import scipy.sparse.linalg as spla

def poisson3d_from_meshgrid_dirichlet(Xc, Yc, Zc, R, u_bc=0.0, tol=1e-10, maxiter=5000):
    """
    Solve ∇²u = R on an equidistant Cartesian grid given by meshgrids Xc,Yc,Zc.
    Dirichlet boundary conditions on the outer boundary (default u=0).

    Parameters
    ----------
    Xc,Yc,Zc : ndarray
        Meshgrid arrays with identical shape (Nx,Ny,Nz) or (Ny,Nx,Nz) etc.
        Must be equidistant along each axis and axis-aligned.
    R : ndarray
        RHS array, same shape as Xc.
    u_bc : float or ndarray or callable
        Dirichlet boundary value.
        - float: constant value on boundary
        - ndarray: same shape, boundary values read from it
        - callable: u_bc(Xc,Yc,Zc) -> array broadcastable to shape
    tol, maxiter : CG solver parameters.

    Returns
    -------
    u : ndarray
        Solution, same shape as R.
    info : int
        CG info (0 = success).
    """
    Xc = np.asarray(Xc)
    Yc = np.asarray(Yc)
    Zc = np.asarray(Zc)
    R  = np.asarray(R)

    if Xc.shape != Yc.shape or Xc.shape != Zc.shape or Xc.shape != R.shape:
        raise ValueError("Xc, Yc, Zc, and R must have the same shape.")

    shape = Xc.shape
    if len(shape) != 3:
        raise ValueError("Expected 3D arrays (shape like (Nx,Ny,Nz)).")

    nx, ny, nz = shape

    # --- infer dx,dy,dz from meshgrid (assumes axis-aligned structured grid) ---
    # We try to detect which axis varies with i/j/k. For typical indexing='ij':
    # X varies with axis 0, Y with axis 1, Z with axis 2.
    # We'll compute spacings robustly from the first line along each axis.
    def _spacing_along_axis(A, axis):
        # take a 1D line along 'axis' while holding others fixed at 0
        sl = [0, 0, 0]
        sl[axis] = slice(None)
        line = A[tuple(sl)]
        if line.size < 2:
            raise ValueError("Need at least 2 points along each axis.")
        d = np.diff(line.astype(float))
        # check equidistant
        if not np.allclose(d, d[0], rtol=1e-10, atol=1e-14):
            raise ValueError("Grid is not equidistant along an axis (or meshgrid not axis-aligned).")
        return float(d[0])

    # Determine which array changes along each axis:
    # Use Xc for dx, Yc for dy, Zc for dz in the common case.
    dx = _spacing_along_axis(Xc, axis=0) if nx > 1 else 1.0
    dy = _spacing_along_axis(Yc, axis=1) if ny > 1 else 1.0
    dz = _spacing_along_axis(Zc, axis=2) if nz > 1 else 1.0

    idx2 = 1.0 / dx**2
    idy2 = 1.0 / dy**2
    idz2 = 1.0 / dz**2

    # boundary mask (outer layer)
    boundary = np.zeros(shape, dtype=bool)
    boundary[0, :, :]  = True
    boundary[-1, :, :] = True
    boundary[:, 0, :]  = True
    boundary[:, -1, :] = True
    boundary[:, :, 0]  = True
    boundary[:, :, -1] = True

    # boundary values array
    if callable(u_bc):
        ub = np.asarray(u_bc(Xc, Yc, Zc), dtype=float)
        ub = np.broadcast_to(ub, shape).copy()
    elif np.isscalar(u_bc):
        ub = np.full(shape, float(u_bc))
    else:
        ub = np.asarray(u_bc, dtype=float)
        if ub.shape != shape:
            raise ValueError("If u_bc is an array, it must have the same shape as R.")

    # We solve for all nodes but enforce u=ub on boundary via operator/b vector.
    # We'll implement A(u) = Laplacian(u) for interior, and A(u)=u on boundary.
    # Then b = R interior, b=ub on boundary.
    b = R.astype(float).copy()
    b[boundary] = ub[boundary]

    def apply_A(u_vec):
        u = u_vec.reshape(shape)

        out = np.empty_like(u)

        # Boundary rows: identity
        out[boundary] = u[boundary]

        # Interior: 7-point Laplacian
        # (use slices to avoid loops)
        out_int = (
            (u[2:, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) * idx2 +
            (u[1:-1, 2:, 1:-1] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) * idy2 +
            (u[1:-1, 1:-1, 2:] - 2*u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) * idz2
        )
        out[1:-1, 1:-1, 1:-1] = out_int

        return out.reshape(-1)

    Aop = spla.LinearOperator((nx*ny*nz, nx*ny*nz), matvec=apply_A, dtype=float)

    # Initial guess: boundary set, interior zero
    u0 = np.zeros(shape, dtype=float)
    u0[boundary] = ub[boundary]

    u_vec, info = spla.cg(Aop, b.reshape(-1), x0=u0.reshape(-1), tol=tol, maxiter=maxiter)
    u = u_vec.reshape(shape)
    return u, info



def dirac_h(r,dirac_type='hat'):

    def hat(r):
        if r<=-1:
            phi = 0
        elif r<=0:
            phi = 1+ r
        elif r<=1:
            phi = 1-r
        else:
            phi = 0
        return phi

    if dirac_type=='hat':
        phi = hat(r)

    return phi


def bubble_heaviside(s1,I,dirac_type='hat',retraction_distance=0.3):

    Xc = s1.Xp
    Yc = s1.Yp
    Zc = s1.Zp

    Rhs = np.zeros_like(Xc)

    b1 = s1.b(1)
    fpx = b1.fp(I,s1)
    fpn = b1.n(I,s1)

    xci = s1.xce
    yci = s1.yce
    zci = s1.zce
    h = xci[5] - xci[4]

    nsup = 2

    for i in range(len(fpx)):
        ni = fpn[:,i]
        xi = fpx[:,i] - ni*h*retraction_distance

        ix = np.round(xi[0]-xci)
        iy = np.round(yi[0]-yci)
        iz = np.round(zi[0]-zci)

        ixi = np.arange(ix-nsup,ix+nsup)
        iyi = np.arange(iy-nsup,iy+nsup)
        izi = np.arange(iz-nsup,iz+nsup)

        for ix1 in ixi:
            for iy1 in iyi:
                for iz1 in izi:
                    Xm = np.array([xci[ix1], yci[iy1], zci[iz1] ])
                    ri = np.linalg.norm(Xm - xi)
                    Rhs[ix1,iy1,iz1] = Rhs[ix1,iy1,iz1] + dirac_h(ri,dirac_type=dirac_tpye) / h

    H, info = poisson3d_from_meshgrid_dirichlet(Xc, Yc, Zc, Rhs, u_bc=0.0)
    print(info)  # 0 means CG converged
    return H
