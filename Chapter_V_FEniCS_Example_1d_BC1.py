from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, dot, TrialFunction, TestFunction
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import basix.ufl

# Création du maillage 1D
domain = mesh.create_interval(MPI.COMM_WORLD, 20, [0.0, 1.0])

# Création de l'espace de fonctions
element = basix.ufl.element("P", domain.basix_cell(), 1)
V = fem.functionspace(domain, element)

# Conditions aux limites
def boundary(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))

dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(0.0, dofs, V)

# Formulation variationnelle
u = TrialFunction(V)
v = TestFunction(V)
f = fem.Constant(domain, 1.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Résolution
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Récupération des coordonnées des noeuds et des valeurs de la solution
x = np.linspace(0.0, 1.0, 21)  # 20 intervalles = 21 noeuds
values = uh.x.array

# Visualisation 1D avec matplotlib
plt.figure()
plt.plot(x, values, 'b-', label="Solution u(x)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Solution du problème 1D")
plt.legend()
plt.grid()
plt.show()
