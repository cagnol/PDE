from fenics import *
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('0', degree=0)

#def boundary(x):
#    return x[0] == 0 or x[1] == 0 or x[0] == 1 or x[1] == 1

def boundary(x):
    tol = 1E-10
    return abs(x[0])<tol or abs(x[1])<tol or abs(x[0]-1)<tol or abs(x[1]-1)<tol

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')
print('error_L2  =', error_L2)

# Hold plot
plt.show()
