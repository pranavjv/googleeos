import numpy as np
import matplotlib.pyplot as plt
import math

# --- Parameters ---
OMEGA_SQ = 1.0      # ω^2 = k/m for SHO (used for Hamiltonian contours & optimization)
M = 1.0             # Mass for Hamiltonian optimization. With M=1, H = 0.5*ω²*x² + 0.5*v²
DT_EULER = 0.1      # Time step for Euler integration
N_STEPS_EULER = 100 # Number of steps for Euler trajectories
LR = 0.26*10   # Learning rate for all
BETA_MOMENTUM = 0.75 # Momentum factor (β)
N_STEPS_OPTIM = 14  # Number of steps for optimization trajectories
PLOT_FILENAME = "sho_phase_space_optimization.png"

# --- Font Size Configuration ---
TITLE_FONTSIZE = 25
AXIS_LABEL_FONTSIZE = 18
AXIS_TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 18
# --- End Font Size Configuration ---

# --- 1. Vector Field Definition (Quartic Kinetic Oscillator) ---
def quartic_kinetic_vector_field(x, p):
    """
    Vector field for L = ½ m x_dot^2 + ¼ α x_dot^4 − ½ k x^2
    
    Parameters:
      x     : position (can be array)
      p     : conjugate momentum (p = m*v + α*v^3) (can be array)
    
    Returns:
      dx_dt : ẋ = real root of α v^3 + m v − p = 0 (array)
      dp_dt : ṗ = −k * x (array)
    """

    m, alpha, k = 1.0, 1.0, 1.0 # System parameters for the quartic oscillator

    # Cardano parameters to solve α v^3 + m v − p = 0 for v (which is dx_dt)
    Q = p / (2 * alpha)
    R_cubed = (m / (3 * alpha)) ** 3

    # Discriminant term: (p/(2*alpha))^2 + (m/(3*alpha))^3
    # This should be non-negative for this cubic form to have one real root.
    sqrt_D_arg = Q * Q + R_cubed
    D_val = np.sqrt(np.maximum(0, sqrt_D_arg)) # Use np.maximum for robustness

    # Cube-root terms for Cardano's solution
    u_arg = Q + D_val
    v_arg = Q - D_val
    
    u = np.cbrt(u_arg) # np.cbrt gives real cube root
    v = np.cbrt(v_arg)

    # ẋ (velocity) is the sum of these two cube roots
    dx_dt = u + v

    # ṗ from the quadratic potential part of Lagrangian
    dp_dt = -k * x

    return dx_dt, dp_dt

# --- 2. RK4 Method for Trajectories ---
def rk4_step(x, p, dt, vector_field_func):
    """Performs a single step of the 4th-order Runge-Kutta method."""
    k1_dx, k1_dp = vector_field_func(x, p)
    
    k2_dx, k2_dp = vector_field_func(x + 0.5 * dt * k1_dx, p + 0.5 * dt * k1_dp)
    
    k3_dx, k3_dp = vector_field_func(x + 0.5 * dt * k2_dx, p + 0.5 * dt * k2_dp)
    
    k4_dx, k4_dp = vector_field_func(x + dt * k3_dx, p + dt * k3_dp)
    
    x_next = x + (dt / 6.0) * (k1_dx + 2*k2_dx + 2*k3_dx + k4_dx)
    p_next = p + (dt / 6.0) * (k1_dp + 2*k2_dp + 2*k3_dp + k4_dp)
    
    return x_next, p_next

def rk4_trajectory(x0, p0, dt, n_steps):
    """Computes a trajectory using RK4 method for the quartic kinetic field."""
    xs = [x0]
    ps = [p0] # Stores momentum values
    for _ in range(n_steps):
        x_next, p_next = rk4_step(xs[-1], ps[-1], dt, quartic_kinetic_vector_field)
        xs.append(x_next)
        ps.append(p_next)
    return np.array(xs), np.array(ps)

# --- 3. Hamiltonian and its Gradient (for SHO system, used in optimization) ---
# H(x,v) = 0.5 * m * v^2 + 0.5 * m * ω^2 * x^2
# Note: The optimization paths use this SHO Hamiltonian, interpreting 'v' as equivalent to 'p' from the plot axis.
def hamiltonian(x, p_or_v, m_param, omega_sq_param):
    """Computes the SHO Hamiltonian. p_or_v is treated as velocity here."""
    return 0.5 * m_param * p_or_v**2 + 0.5 * m_param * omega_sq_param * x**2

def hamiltonian_gradient(x, v, m_param, omega_sq_param):
    """Computes the gradient of the SHO Hamiltonian: (dH/dx, dH/dv)."""
    grad_x = m_param * omega_sq_param * x
    grad_v = m_param * v
    return grad_x, grad_v

# --- 4. Optimizer Step Function Placeholders ---
def gradient_descent_step(x_curr, v_curr, lr, m_param, omega_sq_param):
    """Placeholder: Implement single gradient descent step on Hamiltonian.
    User is expected to fill this function.
    Current implementation is a specific one from user history.
    """
    # grad_x, grad_v = hamiltonian_gradient(x_curr, v_curr, m_param, omega_sq_param)
    # x_next = x_curr - lr * grad_x
    # v_next = v_curr - lr * grad_v
    # return x_next, v_next
    return x_curr - lr * x_curr, 0 # User's specific placeholder logic

def momentum_step(x_curr, v_curr, mom_x_prev, mom_v_prev, lr, beta, m_param, omega_sq_param):
    """
    Placeholder: Implement single momentum step on Hamiltonian.
    User is expected to fill this function.
    Should return: x_next, v_next, new_mom_x, new_mom_v
    """
    # grad_x, grad_v = hamiltonian_gradient(x_curr, v_curr, m_param, omega_sq_param)
    # mom_x_curr = beta * mom_x_prev + lr * grad_x
    # mom_v_curr = beta * mom_v_prev + lr * grad_v
    # x_next = x_curr - mom_x_curr
    # v_next = v_curr - mom_v_curr
    # return x_next, v_next, mom_x_curr, mom_v_curr
    print("Placeholder: momentum_step called. Please implement.")
    # v_next= beta*v_curr - lr* x_curr
    v_next= beta*v_curr -(1-beta)* x_curr
    x_next= x_curr +lr* v_next
    return x_next, v_next, mom_x_prev, mom_v_prev # Return current to avoid error

def dynamic_momentum_step(x_curr, v_curr, mom_x_prev, mom_v_prev, lr, beta, m_param, omega_sq_param, total_steps, current_step):
    """
    Placeholder: Implement single dynamic momentum step on Hamiltonian.
    User is expected to fill this function.
    Learning rate or beta might change based on total_steps and current_step.
    Should return: x_next, v_next, new_mom_x, new_mom_v
    """
    # Example: lr_dynamic = lr * (1 - current_step / total_steps)
    # grad_x, grad_v = hamiltonian_gradient(x_curr, v_curr, m_param, omega_sq_param)
    # mom_x_curr = beta * mom_x_prev + lr_dynamic * grad_x 
    # mom_v_curr = beta * mom_v_prev + lr_dynamic * grad_v
    # x_next = x_curr - mom_x_curr
    # v_next = v_curr - mom_v_curr
    # return x_next, v_next, mom_x_curr, mom_v_curr

    v_next= beta*v_curr -(1-beta)* x_curr
    x_next= x_curr +lr/(1+ 4*v_curr*v_curr)* v_next
    print("Placeholder: dynamic_momentum_step called. Please implement.")
    return x_next, v_next, mom_x_prev, mom_v_prev # Return current to avoid error

# --- Optimizer Trajectory Computation ---
def compute_optimizer_trajectory(x0, v0, optimizer_type, lr, n_steps, m_param, omega_sq_param, beta=BETA_MOMENTUM):
    """Computes an optimization trajectory using specified step function."""
    xs = [x0]
    vs = [v0]

    mom_x, mom_v = 0.0, 0.0 # Initialize for momentum methods

    for i in range(n_steps):
        x_curr, v_curr = xs[-1], vs[-1]
        
        if optimizer_type == "gd":
            x_next, v_next = gradient_descent_step(x_curr, v_curr, lr, m_param, omega_sq_param)
        elif optimizer_type == "momentum":
            x_next, v_next, mom_x, mom_v = momentum_step(x_curr, v_curr, mom_x, mom_v, lr, beta, m_param, omega_sq_param)
        elif optimizer_type == "dynamic_momentum":
            x_next, v_next, mom_x, mom_v = dynamic_momentum_step(x_curr, v_curr, mom_x, mom_v, lr, beta, m_param, omega_sq_param, n_steps, i)
        else:
            print(f"Error: Unknown optimizer_type: {optimizer_type}")
            return np.array(xs), np.array(vs) # Return what we have so far
        
        # Check if step functions are placeholders that might not change x,v
        # or if they indicate non-convergence/error by returning None (optional)
        if x_next is None or v_next is None : # A step function might return None if not properly implemented
             print(f"Warning: Optimizer step function '{optimizer_type}' returned None. Stopping trajectory.")
             break

        xs.append(x_next)
        vs.append(v_next)
        
    return np.array(xs), np.array(vs)

# --- 5. Plotting Function ---
def plot_phase_space_and_optimization(file_name=PLOT_FILENAME):
    """Creates and saves the combined plot."""
    # Define grid for vector field
    imrange = 2.8
    x_range = np.linspace(-imrange, imrange, 20)
    p_range = np.linspace(-imrange, imrange, 20) 
    X_grid, P_grid = np.meshgrid(x_range, p_range) 
    DX_field, DP_field = quartic_kinetic_vector_field(X_grid, P_grid) 

    plt.figure(figsize=(11, 11)) # Increased figure size for external legend

    # Plot vector field (quiver plot)
    plt.quiver(X_grid, P_grid, DX_field, DP_field, color='dimgray', alpha=0.7, 
               label="Vector Field (dx/dt, dv/dt)", headwidth=3, headlength=4, width=0.003)

    # Plot RK4 trajectories for the quartic system
    x_e1, p_e1 = rk4_trajectory(x0=2.0, p0=0.0, dt=DT_EULER, n_steps=N_STEPS_EULER)
    plt.plot(x_e1, p_e1, label="Continuous Trajectories", color='black', linewidth=1.5)
    
    x_e2, p_e2 = rk4_trajectory(x0=0.0, p0=1.5, dt=DT_EULER, n_steps=N_STEPS_EULER)
    plt.plot(x_e2, p_e2,  color='black', linewidth=1.5)

    # # Plot Gradient Descent trajectory on SHO Hamiltonian
    # x_gd, v_gd = compute_optimizer_trajectory(x0=1.5, v0=0.0, optimizer_type="gd", 
    #                                            lr=LR, n_steps=N_STEPS_OPTIM, 
    #                                            m_param=M, omega_sq_param=OMEGA_SQ)
    # if x_gd.size > 0 : plt.plot(x_gd, v_gd, 'o-', label=f"GD on SHO H (lr={LR})", color='orangered', markersize=4, linewidth=1)

    # Plot Momentum trajectory on SHO Hamiltonian
    x_mom, v_mom = compute_optimizer_trajectory(x0=-2.5, v0=0.0, optimizer_type="momentum", 
                                                 lr=LR, n_steps=N_STEPS_OPTIM, 
                                                 m_param=M, omega_sq_param=OMEGA_SQ, beta=BETA_MOMENTUM)
    if x_mom.size > 0 : plt.plot(x_mom, v_mom, 's-', label=f"Momentum", 
                                   color='orange', markersize=7, linewidth=1.5, marker= 'o')

    # Plot Dynamic Momentum trajectory on SHO Hamiltonian (placeholder plot)
    x_dyn_mom, v_dyn_mom = compute_optimizer_trajectory(x0=2.5, v0=0.0, optimizer_type="dynamic_momentum",
                                                         lr=LR, n_steps=N_STEPS_OPTIM,
                                                         m_param=M, omega_sq_param=OMEGA_SQ, beta=BETA_MOMENTUM)
    if x_dyn_mom.size > 0 : plt.plot(x_dyn_mom, v_dyn_mom, 'x-', label=f"VRMomentum",
                                     color='blue', markersize=7, linewidth=1.5, marker='o')

    # Plot SHO Hamiltonian contours (using P_grid as the velocity component for H)
    #H_contours = hamiltonian(X_grid, P_grid, M, OMEGA_SQ)
    #plt.contour(X_grid, P_grid, H_contours, levels=10, colors='gold', alpha=0.6, linestyles='dotted', linewidths=1)

    plt.xlabel("Position (x)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Velocity (v)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f"Phase Space Diagram with quadratic potential", fontsize=TITLE_FONTSIZE)
    plt.xticks(fontsize=AXIS_TICK_FONTSIZE)
    plt.yticks(fontsize=AXIS_TICK_FONTSIZE)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    
    # Set limits and aspect ratio BEFORE placing legend and adjusting subplots
    plt.xlim(x_range.min(), x_range.max())
    plt.ylim(p_range.min(), p_range.max())
    plt.axis('equal') 

    # Adjust legend to be outside the plot using user's specified anchor
    plt.legend(loc='lower left' ,bbox_to_anchor=(0.19, -0.35), fontsize=LEGEND_FONTSIZE)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust subplot to make room for legend
    # These values (left, bottom, right, top) are fractions of the figure width and height.
    # Increase left and bottom padding to make space for a legend anchored far to the left/bottom.
    plt.subplots_adjust(left=0.3, bottom=0.25, right=0.95, top=0.9)

    plt.savefig(file_name)
    plt.close()
    print(f"Plot saved to {file_name}")

# --- Main execution ---
if __name__ == "__main__":
    plot_phase_space_and_optimization() 