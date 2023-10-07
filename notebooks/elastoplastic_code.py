import os
from dolfin import *
import numpy as np
parameters["form_compiler"]["representation"] = 'quadrature'
parameters["form_compiler"]["representation"] = "tsfc"
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

msh_files_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/msh"
dat_files_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/dat"
xdmf_files_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/xdmf"
plots_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/plot"

msh_files = os.listdir(msh_files_folder)
dat_files = os.listdir(dat_files_folder)

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
    return out_mesh

def extract_fourth_column(dat_file):
    fourth_column = []
    with open(dat_file, "r") as file:
        for line in file:
            elements = line.strip().split()
            fourth_column.append(int(elements[3])) 
    return fourth_column

for msh_file, dat_file in zip(msh_files, dat_files):
   
    msh_path = os.path.join(msh_files_folder, msh_file)
    dat_path = os.path.join(dat_files_folder, dat_file)
    
    import meshio
    msh_path = os.path.join(msh_files_folder, msh_file)
    mesh_from_file = meshio.read(msh_path)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    xdmf_path = os.path.join(xdmf_files_folder, f"{msh_file[:-4]}.xdmf")
    meshio.write(xdmf_path, triangle_mesh)
    mesh = Mesh()
    with XDMFFile(xdmf_path) as infile:
        infile.read(mesh)
    
    x_scale = 1.0 / 1023.0
    y_scale = 1.0 / 1023.0
    for vertex in mesh.coordinates():
        vertex[0] *= x_scale
        vertex[1] *= y_scale

    fourth_column_array = extract_fourth_column(dat_path)

    materials = MeshFunction('double', mesh, 2)
    G = VectorFunctionSpace(mesh, "DG", 0)
    g = Function(G)
    local_values_material = np.zeros_like(g.vector().get_local())
    for cell in cells(mesh):
        midpoint = cell.midpoint().array()
        i = (midpoint[0])
        j = (midpoint[1])
        local_values_material[cell.index()] = fourth_column_array[cell.index()]
        materials[cell] = int(local_values_material[cell.index()])    
    g.vector().set_local(local_values_material)
    XDMFFile(MPI.comm_world, xdmf_path).write_checkpoint(g,"g",0)

    class al(UserExpression):
        def __init__(self, materials, al0, al1, **kwargs):
            super().__init__(**kwargs)
            self.materials = materials
            self.k_0 = al0
            self.k_1 = al1
        def eval_cell(self, values, x, cell):
            if self.materials[cell.index] == 0:
                values[0] = self.k_0
            else:
                values[0] = self.k_1
        def value_shape(self):
            return () 
    
    E1 = 69e3 
    nu1 = 0.33
    sig01 = 276 
    E2 = 454e3
    nu2 = 0.25
    sig02 = 620 

    E = al(materials, E1, E2, degree = 0)
    nu = al(materials, nu1, nu2, degree = 0)
    sig0 = al(materials, sig01, sig02, degree = 0)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    Et = E/100.  
    H = E*Et/(E-Et)  

    deg_u = 2
    deg_stress = 2
    V = VectorFunctionSpace(mesh, "CG", deg_u)
    We = VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
    W = FunctionSpace(mesh, We)
    W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W0 = FunctionSpace(mesh, W0e)
    sig = Function(W)
    sig_old = Function(W)
    n_elas = Function(W)
    beta = Function(W0)
    p = Function(W0, name="Cumulative plastic strain")
    u = Function(V, name="Total displacement")
    du = Function(V, name="Iteration correction")
    Du = Function(V, name="Current increment")
    v = TrialFunction(V)
    u_ = TestFunction(V)
    P0 = FunctionSpace(mesh, "DG", 0)
    p_avg = Function(P0, name="Plastic strain")

    class bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0., 1e-8)
    
    class top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1., 1e-8)

    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    bottom().mark(boundaries, 1)
    top().mark(boundaries, 2)
    loading = Expression("t", t=0, degree=1)
    bc_bottom = DirichletBC(V.sub(1), Constant((0.)), boundaries, 1)
    ds = Measure("ds")(subdomain_data=boundaries)
    n = FacetNormal(mesh)
    bc_u = [bc_bottom]

    def F_ext(v):
        return loading*dot(n, v)*ds(2)

    def eps(v):
        e = sym(grad(v))
        return as_tensor([[e[0, 0], e[0, 1], 0],
                      [e[0, 1], e[1, 1], 0],
                      [0, 0, 0]])

    def sigma(eps_el):
        return lmbda*tr(eps_el)*Identity(3) + 2*mu*eps_el

    def as_3D_tensor(X):
        return as_tensor([[X[0], X[3], 0],
                      [X[3], X[1], 0],
                      [0, 0, X[2]]])

    ppos = lambda x: (x+abs(x))/2.
    def proj_sig(deps, old_sig, old_p):
        sig_n = as_3D_tensor(old_sig)
        sig_elas = sig_n + sigma(deps)
        s = dev(sig_elas)
        sig_eq = sqrt(3/2.*inner(s, s))
        f_elas = sig_eq - sig0 - H*old_p
        dp = ppos(f_elas)/(3*mu+H)
        n_elas = s/sig_eq*ppos(f_elas)/f_elas
        beta = 3*mu*dp/sig_eq
        new_sig = sig_elas-beta*s
        return as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
               as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
               beta, dp

    def sigma_tang(e):
        N_elas = as_3D_tensor(n_elas)
        return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*inner(N_elas, e)*N_elas-2*mu*beta*dev(e)

    def local_project(v, V, u=None):
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dxm
        b_proj = inner(v, v_)*dxm
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        if u is None:
            u = Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)
            return
    
    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
    dxm = dx(metadata=metadata)

    a_Newton = inner(eps(v), sigma_tang(eps(u_)))*dxm
    res = -inner(eps(u_), as_3D_tensor(sig))*dxm + F_ext(u_)

    sig_n1 = as_3D_tensor(sig)
    s1 = dev(sig_n1)
    Vs = FunctionSpace(mesh, "DG", 0)
    sig_eq1 = local_project(sqrt(3/2.*inner(s1, s1)), Vs)
    vm = sig_eq1.vector().max()
    print("max_vonmises: ",vm)

    Nitermax, tol = 200, 1e-8  
    Nincr = 100
    load_steps = np.linspace(0, 500., Nincr+1)[1:]
    results = np.zeros((Nincr+1, 2))

    for (i, t) in enumerate(load_steps):
    
        sig_n1 = as_3D_tensor(sig)
        s1 = dev(sig_n1)
        Vs = FunctionSpace(mesh, "DG", 0)
        sig_eq1 = local_project(sqrt(3/2.*inner(s1, s1)), Vs)
        vm = sig_eq1.vector().max()
        print("vm: ",vm)
        
        loading.t = t
        A, Res = assemble_system(a_Newton, res, bc_u)
        nRes0 = Res.norm("l2")
        nRes = nRes0
        Du.interpolate(Constant((0, 0)))
        print("Increment:", str(i+1))
        niter = 0
        while nRes/nRes0 > tol and niter < Nitermax:
            solve(A, du.vector(), Res, "umfpack")
            Du.assign(Du+du)
            deps = eps(Du)
            sig_, n_elas_, beta_, dp_ = proj_sig(deps, sig_old, p)
            local_project(sig_, W, sig)
            local_project(n_elas_, W, n_elas)
            local_project(beta_, W0, beta)
        
            A, Res = assemble_system(a_Newton, res, bc_u)
            nRes = Res.norm("l2")
            print("    Residual:", nRes)
            niter += 1
        u.assign(u+Du)
        p.assign(p+local_project(dp_, W0))
        sig_old.assign(sig)
        p_avg.assign(project(p, P0))
        k = p_avg.vector().max()
        print("max_plastic_strain: ", k)
        if k >= 0.05:
            x= P0.tabulate_dof_coordinates()
            max_index = np.argmax(p_avg.vector()[:])
            print("Max_plastic_strain (in 1023. x 1023.) at:", 1023.*x[max_index])
            break
        results[i+1, :] = (u(0.5, 1.)[1], t)

    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    k = plot(p_avg, mode='color')
    plt.colorbar(k)
    plt.title(r"Plastic Strain",fontsize=26)
    plot_path = os.path.join(plots_folder, f"{msh_file[:-4]}_plasticstrain.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    sig_n1 = as_3D_tensor(sig)
    s1 = dev(sig_n1)
    Vs = FunctionSpace(mesh, "DG", 1)
    sig_eq1 = local_project(sqrt(3/2.*inner(s1, s1)), Vs)
    p = plot(sig_eq1, mode='color')
    plt.colorbar(p)
    plt.title(r"Von Mises",fontsize=26)
    plot_path = os.path.join(plots_folder, f"{msh_file[:-4]}_vonmises.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    plt.plot(results[:, 0], results[:, 1], "-o")
    plt.xlabel("Displacement of upper boundary (in mm)")
    plt.ylabel(r"Applied load (in N)")
    plt.grid(True)
    plot_path = os.path.join(plots_folder, f"{msh_file[:-4]}_loadplot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    p = plot(u, mode = 'displacement')
    plt.colorbar(p)
    plt.title(r"Total Displacement",fontsize=26)
    plot_path = os.path.join(plots_folder, f"{msh_file[:-4]}_total_disp.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
