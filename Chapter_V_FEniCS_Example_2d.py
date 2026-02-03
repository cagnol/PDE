import numpy as np
import ufl
from dolfinx import fem, mesh, plot
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
import matplotlib.pyplot as plt

# creation du mesg
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)
V = fem.functionspace(domain, ("P", 1))

# conditions aux limites
u_D = fem.Constant(domain, 0.0)

# creation des limites
def boundary_facets(x):
    return np.logical_or.reduce((
        np.isclose(x[0], 0.0),
        np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0),
        np.isclose(x[1], 1.0)
    ))

facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, boundary_facets)
dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
bc = fem.dirichletbc(u_D, dofs, V)

# Definition du probleme variationel
u = ufl.TrialFunction(V)    # Placeholder, u NE CONTIENT PAS LA SOLUTION (meme apres problem.solve())
v = ufl.TestFunction(V)     # fonction de test
f = fem.Constant(domain, 1.0)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# cree un probleme lineaire et le resout
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()  # uh: contient la solution

try: # tente de visualiser la solution avec pyvista
    import pyvista
    pyvista.set_jupyter_backend("static")
    plotter = pyvista.Plotter()

    # Create grid from mesh
    topology, cell_types, geometry = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach solution values
    grid.point_data["u"] = uh.x.array
    grid.set_active_scalars("u")

    # Plot
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.show()
except ImportError as i:
    print(i.__repr__()) # utile pour debug
    print("PyVista not available for visualization")

# Calcul de l'erreur
error_form = fem.form(ufl.inner(uh, uh) * ufl.dx)
error_local = fem.assemble_scalar(error_form)
error_l2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

print('error_L2 =', error_l2)