# signal_gen_1d.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Literal
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- FEM / dolfinx imports ----
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
import ufl
from dolfinx.fem.petsc import assemble_matrix

# =========================================================
# Helper: Sparse Matrix Conversion
# =========================================================
def _petsc_to_scipy(petsc_mat) -> sp.csr_matrix:
    """Convert PETSc matrix to Scipy CSR."""
    row, col, val = petsc_mat.getValuesCSR()
    return sp.csr_matrix((val, col, row), shape=petsc_mat.getSize())

def _split_counts_by_prop(n_total: int, props: List[float]) -> List[int]:
    props = np.asarray(props, dtype=float)
    props = props / props.sum()
    base = np.floor(n_total * props).astype(int)
    rem = n_total - base.sum()
    frac = (n_total * props) - np.floor(n_total * props)
    order = np.argsort(-frac)
    for i in range(rem):
        base[order[i % len(base)]] += 1
    return base.tolist()

# =========================================================
# Optimized Solver Class (Caches Mesh & Matrices)
# =========================================================
class MaternSolver1D:
    def __init__(self, n_points: int, domain: Tuple[float, float] = (-0.5, 0.5)):
        """
        Initializes the mesh and pre-assembles basic matrices (Mass and Stiffness).
        This avoids recompilation inside the sampling loop.
        """
        self.n_points = n_points
        self.domain = domain
        
        # 1. Build Mesh ONCE
        # Create unit interval first
        self.mesh = mesh.create_unit_interval(MPI.COMM_WORLD, n_points - 1)
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
        
        # 2. Map coordinates to target domain
        x0, x1 = domain
        self.mesh.geometry.x[:, 0] = x0 + self.mesh.geometry.x[:, 0] * (x1 - x0)
        
        # 3. Identify Boundary Dofs (Dirichlet BCs)
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        
        def boundary(x):
            return np.isclose(x[0], x0) | np.isclose(x[0], x1)
            
        boundary_facets = mesh.locate_entities_boundary(self.mesh, tdim - 1, boundary)
        # Find DOFs associated with boundary facets
        self.bc_dofs = fem.locate_dofs_topological(self.V, tdim - 1, boundary_facets)
        
        # 4. Pre-assemble Matrices (Mass M and Stiffness K) WITHOUT BCs
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        # Mass form
        a_m = u * v * ufl.dx
        M_petsc = assemble_matrix(fem.form(a_m))
        M_petsc.assemble()
        self.M_sp = _petsc_to_scipy(M_petsc)
        
        # Stiffness form (grad u . grad v)
        a_k = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        K_petsc = assemble_matrix(fem.form(a_k))
        K_petsc.assemble()
        self.K_sp = _petsc_to_scipy(K_petsc)
        
        # 5. Pre-calculate Noise Scaling (Mass Lumping)
        # We need M^(1/2) * xi. Using lumped mass approximation for speed.
        self.m_lumped_diag = np.array(self.M_sp.sum(axis=1)).flatten()
        self.m_sqrt_diag = np.sqrt(np.maximum(self.m_lumped_diag, 0.0))
        
        # Extract x coordinates for return
        x_coords = self.mesh.geometry.x[:, 0]
        self.sort_idx = np.argsort(x_coords)
        self.x_grid = x_coords[self.sort_idx]
        
    def sample(self, nu: float, kappa: float, sigma: float) -> np.ndarray:
        """
        Generates a sample using cached matrices.
        A = kappa^2 * M + K
        """
        ndofs = self.M_sp.shape[0]
        
        # 1. Construct Operator A using sparse arithmetic
        A_sp = (kappa**2) * self.M_sp + self.K_sp
        
        # 2. Apply Dirichlet BCs manually to the Sparse Matrix
        if len(self.bc_dofs) > 0:
            # Fast implementation:
            mask = np.ones(ndofs, dtype=bool)
            mask[self.bc_dofs] = False
            
            # Diagonal matrix P: 1 for inner nodes, 0 for bc nodes
            P = sp.diags(mask.astype(float))
            
            # A_constrained = P * A * P + (I - P)
            # This sets BC rows/cols to 0 and diagonal to 1.
            # For Dirichlet homogeneous, this implies u_bc = 0.
            
            I_minus_P = sp.diags((~mask).astype(float))
            A_final = P @ A_sp @ P + I_minus_P
        else:
            A_final = A_sp

        # 3. Generate RHS Noise
        xi = np.random.randn(ndofs)
        rhs = self.m_sqrt_diag * xi  # M^(1/2) * xi
        
        # Apply BC to RHS (Force to 0)
        rhs[self.bc_dofs] = 0.0
        
        # 4. Solve
        alpha = int(round(nu + 0.5))
        
        # Pre-factorize A
        try:
            solver = spla.splu(A_final)
        except RuntimeError:
            # Fallback if matrix is singular (rare with shift)
            solver = spla.splu(A_final + sp.eye(ndofs)*1e-6)

        xvec = rhs
        
        for i in range(alpha):
            if i > 0:
                # Apply M operator for subsequent steps
                xvec = self.M_sp @ xvec
                xvec[self.bc_dofs] = 0.0
            
            xvec = solver.solve(xvec)
            
        # 5. Scaling & Return
        xvec *= sigma
        return xvec[self.sort_idx]


# =========================================================
# Top Level Generation
# =========================================================
def generate_spde_true_signals_by_classes(
    *,
    n_total: int,
    n_points: int,
    domain: Tuple[float, float] = (-0.5, 0.5),
    classes: List[Dict] = None,
    sigma: float = 1.0,
    seed: Optional[int] = None,
    save_previews: bool = False,
    preview_dir: str = "true_signal_previews",
) -> Dict[str, np.ndarray]:
    
    assert classes and len(classes) > 0, "classes cannot be empty"
    if seed is not None:
        np.random.seed(int(seed))

    # Initialize the Solver ONCE
    print("Initializing Finite Element Solver...")
    solver = MaternSolver1D(n_points=n_points, domain=domain)
    
    props = [float(c["prop"]) for c in classes]
    counts = _split_counts_by_prop(n_total, props)

    all_u = []
    all_nu = []
    all_kappa = []
    all_cid = []

    # Prepare Task List
    tasks = []
    for cid, (c, n_i) in enumerate(zip(classes, counts)):
        if n_i == 0: continue
        nu_i = float(c["nu"])
        kmin, kmax = map(float, c["kappa"])
        
        a_beta, b_beta = 1.0, 1.0 #5.0, 2.0 #
        beta_draws = np.random.beta(a_beta, b_beta, size=n_i)
        kappas_i = kmin + (kmax - kmin) * beta_draws
        
        for idx, k_val in enumerate(kappas_i):
            tasks.append({
                "cid": cid,
                "nu": nu_i,
                "kappa": k_val,
                "sample_idx": idx # Track index for unique filenames
            })

    # Execute Sampling with Progress Bar
    print(f"Generating {n_total} signals...")
    for task in tqdm(tasks, total=len(tasks), desc="Sampling SPDE"):
        cid = task["cid"]
        nu_i = task["nu"]
        kappa_j = float(task["kappa"])
        s_idx = task["sample_idx"]

        # Call the fast solver
        u_vec = solver.sample(nu=nu_i, kappa=kappa_j, sigma=sigma)
        
        # Normalize (Max-Abs)
        m = np.max(np.abs(u_vec))
        if m > 1e-12:
            u_vec = u_vec / m

        all_u.append(u_vec)
        all_nu.append(nu_i)
        all_kappa.append(kappa_j)
        all_cid.append(cid)

        # Preview
        if save_previews:
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir, exist_ok=True)
            
            plt.figure(figsize=(8, 3))
            plt.plot(solver.x_grid, u_vec, lw=1.5)
            plt.axvline(domain[0], color="k", ls="--", lw=0.8)
            plt.axvline(domain[1], color="k", ls="--", lw=0.8)
            plt.title(f"Class {cid}: nu={nu_i}, kappa={kappa_j:.2f}")
            plt.xlabel("x"); plt.ylabel("u(x)")
            plt.tight_layout()
            fname = f"class{cid}_sample{s_idx}_nu{nu_i}.png"
            plt.savefig(os.path.join(preview_dir, fname), dpi=100)
            plt.close()

    y_true = np.vstack(all_u).astype(float)
    return {
        "y_true": y_true,
        "x_grid": solver.x_grid,
        "nu_used": np.asarray(all_nu, dtype=float),
        "kappa_used": np.asarray(all_kappa, dtype=float),
        "class_id": np.asarray(all_cid, dtype=int),
    }

# =========================================================
# Execution Block
# =========================================================
if __name__ == "__main__":
    classes = [
        {"nu": 0.5, "prop": 0.5, "kappa": [1, 20]},
        {"nu": 1.5, "prop": 0.3, "kappa": [1, 25]},
        {"nu": 2.5, "prop": 0.2, "kappa": [1, 30]},
    ]
    
    out = generate_spde_true_signals_by_classes(
        n_total=1024, 
        n_points=64, 
        domain=(-1, 1), 
        classes=classes, 
        save_previews=False
    )
    
    save_filename = "./data/signal_1024.npz"
    np.savez_compressed(save_filename, **out)
    
    print("Generation Complete.")
    print(f"Data saved to: {save_filename}")
    print("Shapes:", out["y_true"].shape)
    
    print("Keys in npz:", list(out.keys()))