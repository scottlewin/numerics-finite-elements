



mu = 10000
u = -10

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

# --- FEM Helper Functions ---

def ref_shape_functions(xi):
    xi = np.atleast_2d(xi)
    return np.array([1 - xi[:,0] - xi[:,1], xi[:,0], xi[:,1]]).T

def derivative_of_ref_shape_functions():
    return np.array([[-1, -1], [1, 0], [0, 1]])

def jacobian(node_coords):
    dN_dxi = derivative_of_ref_shape_functions()
    return np.dot(dN_dxi.T, node_coords)

def global_x_of_xi(xi, global_node_coords):
    N = ref_shape_functions(xi)
    return np.dot(N, global_node_coords)

def det_jacobian(J):
    return np.linalg.det(J)

def inverse_jacobian(J):
    return np.linalg.inv(J)

def global_deriv(J):
    J_inv = inverse_jacobian(J)
    dN_dxi = derivative_of_ref_shape_functions()
    return dN_dxi @ J_inv


def mass_matrix_for_element(node_coords):
    # Define the quadrature points for a linear triangular element (same as used for stiffness)
    xi_eta = np.array([[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]])  # Quadrature points in reference space
    mass_matrix = np.zeros((3, 3))  # Initialize mass matrix for the element
    J = jacobian(node_coords)  # Jacobian matrix for the transformation
    detJ = det_jacobian(J)  # Determinant of the Jacobian
    
    # Loop over quadrature points
    for q in range(3):
        xi = xi_eta[q, 0]
        eta = xi_eta[q, 1]
        
        # Evaluate shape functions at the quadrature point
        N = ref_shape_functions(np.array([[xi, eta]]))[0]
        
        # Integrate shape functions at quadrature points to form the mass matrix
        for i in range(3):
            for j in range(3):
                mass_matrix[i, j] += N[i] * N[j] * detJ * 1/6 
    
    return mass_matrix

def stiffness_advection(global_node_coords):
    ref_xi = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]  # Quadrature points
    J = jacobian(global_node_coords)
    detJ = det_jacobian(J)
    dN_dxdy = global_deriv(J)
    N = ref_shape_functions(ref_xi)
    
    total_advection = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            total_advection[i, j] = N[i, 0] * dN_dxdy[j, 1] + N[i, 1] * dN_dxdy[j, 1] + N[i, 2] * dN_dxdy[j, 1]

    # Compute the diffusion and advection terms (scaled by physical parameters)
    mu_term = 1/2 * detJ * (dN_dxdy @ dN_dxdy.T)
    u_term = 1/6 * total_advection * detJ
    
    return mu * mu_term - u * u_term

def force_vector_for_element(global_node_coords, S):
    J = jacobian(global_node_coords)
    ref_xi = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]
    N = ref_shape_functions(ref_xi)
    global_coords = global_x_of_xi(ref_xi, global_node_coords)
    detJ = det_jacobian(J)
    S_vals = np.zeros((3, 1))
    for i in range(3):
        S_vals[i] += S(global_coords[i])
    return np.sum(1/6 * S_vals * N, axis=0) * detJ


def euler_step(M, K, F, Psi, dt):
    """
    Perform a single Euler step for the time integration of the first-order system:
    M * dPsi/dt + K * Psi = F

    Parameters:
        M: Mass matrix (global)
        K: Stiffness matrix (global)
        F: Force vector (global)
        Psi: Solution vector (state variable)
        dt: Time step

    Returns:
        Psi_new: Solution vector after one time step
    """
    # Compute dPsi/dt at the current step
    dpsidt = spsolve(M, F - K @ Psi)
    
    # Update Psi using Euler's method
    Psi_new = Psi + dt * dpsidt
    
    return Psi_new

def find_element_and_interpolate(global_node_coords, IEN, Psi, point):
    """
    Find the element containing the given point and compute the value of Psi at that point.

    Parameters:
        global_node_coords: Array of shape (n_nodes, 2) with the coordinates of all nodes in the global system.
        IEN: Array of shape (n_elements, n_nodes_per_element), where each row contains the indices
                      of the nodes defining an element.
        Psi: Array of values of Psi at the global nodes.
        point: Coordinates of the point to evaluate, given as (x, y).

    Returns:
        Psi_at_point: The interpolated value of Psi at the given point.
    """
    def is_point_in_element(global_coords, element_coords, point):
        """
        Check if a point is inside a triangular element using barycentric coordinates.
        """
        # Extract node coordinates for the element
        x1, y1 = element_coords[0]
        x2, y2 = element_coords[1]
        x3, y3 = element_coords[2]
        xp, yp = point

        # Compute areas for barycentric coordinates
        detT = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        alpha = ((x2 - xp) * (y3 - yp) - (x3 - xp) * (y2 - yp)) / detT
        beta = ((x3 - xp) * (y1 - yp) - (x1 - xp) * (y3 - yp)) / detT
        gamma = 1 - alpha - beta

        # Check if point lies inside the element
        return (alpha >= 0) and (beta >= 0) and (gamma >= 0), np.array([alpha, beta, gamma])

    # Loop through all elements to find the one containing the point
    for elem_idx, element in enumerate(IEN):
        element_coords = global_node_coords[element]  # Get global coordinates of the element's nodes

        # Check if the point lies within this element
        inside, bary_coords = is_point_in_element(global_node_coords, element_coords, point)
        if inside:
            # Interpolate Psi using the barycentric coordinates
            element_Psi = Psi[element]  # Values of Psi at the element's nodes
            Psi_at_point = np.dot(bary_coords, element_Psi)
            return Psi_at_point

    # If no element is found, raise an error
    raise ValueError("Point is outside the mesh or not in any element.")
    


def solve_fem_euler(nodes, IEN, ID, boundary_nodes, T_max, dt, S):
    """
    Solve the FEM problem using Euler time-stepping.

    Parameters:
        nodes: Node coordinates.
        IEN: Element connectivity array.
        ID: Mapping of node indices to global equation indices.
        boundary_nodes: List of boundary node indices.
        T_max: Maximum simulation time.
        dt: Time step size.
        S: Source term function.

    Returns:
        psi_nodes_list: List of Psi values for each time step.
    """
    # Initialize sparse matrices
    N_equations = np.max(ID) + 1
    K = sparse.lil_matrix((N_equations, N_equations))  # Global stiffness matrix
    M = sparse.lil_matrix((N_equations, N_equations))  # Global mass matrix
    F = np.zeros(N_equations)  # Global force vector

    # Assemble loop over elements
    for e in range(IEN.shape[0]):
        node_coords = nodes[IEN[e, :], :]
        
        # Assemble the element stiffness matrix and mass matrix
        k_e = stiffness_advection(node_coords)
        m_e = mass_matrix_for_element(node_coords)

        # Assemble the global stiffness matrix and mass matrix (using sparse matrix assembly)
        for a in range(3):
            for b in range(3):
                A = ID[IEN[e, a]]
                B = ID[IEN[e, b]]
                if A >= 0 and B >= 0:
                    K[A, B] += k_e[a, b]
                    M[A, B] += m_e[a, b]

        # Assemble the force vector (assuming S is the source term)
        f_e = force_vector_for_element(node_coords, S)
        for a in range(3):
            A = ID[IEN[e, a]]
            if A >= 0:
                F[A] += f_e[a]

    # Convert sparse matrices to CSR format for efficient operations
    K = K.tocsr()
    M = M.tocsr()

    # Initial condition (assuming zero initial displacement and velocity)
    Psi = np.zeros(N_equations)  # Initial displacement (solution vector)

    # Create a list containing the full solution for each time step
    psi_nodes_list = []
    pollution_list = []
    # Time-stepping loop (Euler)
    t = 0
    tlist = []
    while t < T_max:
        # Euler time-stepping
        if t > 28800:
            F = 0
        else: F = F
        Psi = euler_step(M, K, F, Psi, dt)

        # Interpolate Psi values back to all nodes for visualization
        Psi_nodes = np.zeros(len(nodes))
        for i, node_id in enumerate(ID):
            if node_id >= 0:  # Ensure we only use valid node indices
                Psi_nodes[i] = Psi[node_id]
        pollution_list.append(find_element_and_interpolate(nodes, IEN, Psi_nodes, (440000, 171625)))
        psi_nodes_list.append(Psi_nodes)
        tlist.append(t)

        # Update time
        t += dt

        # Visualization (optional)
        if t % 1000 < dt:
            plt.triplot(nodes[:, 0], nodes[:, 1], IEN)
            plt.plot(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], 'ro')
            plt.tripcolor(nodes[:, 0], nodes[:, 1], Psi_nodes, shading='flat', triangles=IEN)
            plt.title(f"Time = {t:.2f}")
            plt.scatter(442365, 115483, color='black')
            plt.scatter(473993, 171625, color='pink')
            plt.scatter(440000, 171625, color='orange')
            plt.colorbar()
            plt.show()
    plt.plot(tlist, pollution_list)

    return psi_nodes_list

def closest_node_to_point_optimized(nodes, point):
    """
    Finds the closest node to a given point using a k-d tree for efficient search.

    Parameters:
    - nodes: 2D array (N x 2) where each row contains the (x, y) coordinates of a node.
    - point: 1D array of length 2 containing the (x, y) coordinates of the point of interest.

    Returns:
    - index: The index of the node closest to the point.
    - closest_node: The coordinates (x, y) of the closest node.
    """
    # Build the k-d tree from the nodes
    tree = cKDTree(nodes)

    # Query the tree for the closest point
    distance, index = tree.query(point)

    # Get the coordinates of the closest node
    closest_node = nodes[index]
    
    return index, closest_node




# Set up grid
nodes = np.loadtxt('griddata/las_nodes_10k.txt')
IEN = np.loadtxt('griddata/las_IEN_10k.txt', dtype=np.int64)
boundary_nodes = np.loadtxt('griddata/las_bdry_10k.txt', dtype=np.int64)
ID = np.zeros(len(nodes), dtype=np.int64)
n_eq = 0
for i in range(len(nodes[:, 1])):
    if i in boundary_nodes:
        ID[i] = -1 
    else: 
        ID[i] = n_eq
        n_eq += 1 

def S_local1(x):
    S = np.exp(-(((x[0] - 442365) ** 2 + (x[1] - 115483) ** 2)) / 10000)
    return S

def S_local(x, sigma=5000, mu_x=442365, mu_y=115483):
    # Compute the squared distance from the center (mu_x, mu_y)
    dist_squared = (x[0] - mu_x) ** 2 + (x[1] - mu_y) ** 2
    
    # Normalized Gaussian (2D)
    A = 1 / (2 * np.pi * sigma**2)  # Normalization factor
    S = A * np.exp(-dist_squared / (2 * sigma**2))  # Normalized 2D Gaussian
    return S
# Solve the FEM problem with time stepping
#solve_fem(nodes, IEN, ID, boundary_nodes, T_max=8000.0, dt=5, S=S_local)
psi_list = solve_fem_euler(nodes, IEN, ID, boundary_nodes, T_max=50000.0, dt=5, S=S_local)



value_at_reading = find_element_and_interpolate(nodes, IEN, psi_list[-1], (473993, 171625) )
value_at_otherbit = find_element_and_interpolate(nodes, IEN, psi_list[-1], (430000, 171625))
value_at_southampton = find_element_and_interpolate(nodes, IEN, psi_list[-1],(442365, 115483) )
print(value_at_reading, value_at_otherbit, value_at_southampton)
otherbit = closest_node_to_point_optimized(nodes, (440000, 171625))
reading = closest_node_to_point_optimized(nodes, (473993, 171625))

