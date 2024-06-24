import jax
import jax.numpy as jnp
from jax import random, jit, vmap


def _euler_step(f, x, t, h):
    """
    Perform a single step of the Euler integration.

    Parameters:
    - f: The derivative function of the system, x' = f(x, t)
    - x: Current value of the dependent variable
    - t: Current time
    - h: Step size

    Returns:
    - x_next: Value of x at t + h
    """
    x_next = x + h * f(x, t)
    return x_next

def _rk4_step(f, x, t, h):
    """
    Perform a single step of the fourth-order Runge-Kutta integration.

    Parameters:
    - f: The derivative function of the system, x' = f(x, t)
    - x: Current value of the dependent variable
    - t: Current time
    - h: Step size

    Returns:
    - x_next: Value of x at t + h
    """
    k1 = f(x, t)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(x + h * k3, t + h)
    x_next = x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next


def _solve_ode(f, x0, t0, tf, h):
    """
    Solve an ODE using the RK4 method.

    Parameters:
    - f: The derivative function of the system, x' = f(x, t)
    - x0: Initial condition
    - t0: Initial time
    - tf: Final time
    - h: Step size

    Returns:
    - xs: Array of x values from t0 to tf
    """
    num_steps = int((tf - t0) / h)
    xs = jnp.zeros((num_steps + 1,) + x0.shape)
    xs = xs.at[0].set(x0)
    t = t0

    for i in range(num_steps):
        xs = xs.at[i + 1].set(_rk4_step(f, xs[i], t, h))
        t += h

    return xs


def _euler_step_treemapped(f, x, t, h):
    """
    Perform a single step of the Euler integration.

    Parameters:
    - f: The derivative function of the system, x' = f(x, t)
    - x: Current value of the dependent variable
    - t: Current time
    - h: Step size

    Returns:
    - x_next: Value of x at t + h
    """
    xdt = f(x, t)
    x_next = jax.tree_map(lambda x, xdt: x + h * xdt, x, xdt)
    return x_next


def _rk4_step_treemapped(f, x, t, h):
    """
    Perform a single step of the fourth-order Runge-Kutta integration.

    Parameters:
    - f: The derivative function of the system, x' = f(x, t)
    - x: Current value of the dependent variable
    - t: Current time
    - h: Step size

    Returns:
    - x_next: Value of x at t + h
    """
    k1 = f(x, t)
    k2 = f(jax.tree_map(lambda x, k1: x + 0.5 * h * k1, x, k1), t + 0.5 * h)
    k3 = f(jax.tree_map(lambda x, k2: x + 0.5 * h * k2, x, k2), t + 0.5 * h)
    k4 = f(jax.tree_map(lambda x, k3: x + h * k3, x, k3), t + h)
    x_next = jax.tree_map(lambda x, k1, k2, k3, k4: x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4), x, k1, k2, k3, k4)
    return x_next


def _solve_latent_ode(f, latents, t0, tf, h, method="rk4", stop_gradient=False):
    """
    Solve an ODE using the RK4 method.

    Parameters:
    - f: The derivative function of the system, x' = f(x, t)
    - x0: Initial condition
    - t0: Initial time
    - tf: Final time
    - h: Step size

    Returns:
    - xs: Array of x values from t0 to tf
    """
    num_steps = int((tf - t0) / h)

    # Unpack latents
    p, a, window = latents

    # Define latent trajectory
    p_traj = jnp.zeros((num_steps + 1,) + p.shape)
    a_traj = jnp.zeros((num_steps + 1,) + a.shape)
    window_traj = jnp.zeros((num_steps + 1,) + window.shape)

    # Set initial state
    p_traj = p_traj.at[0].set(p)
    a_traj = a_traj.at[0].set(a)
    window_traj = window_traj.at[0].set(window)

    t = t0
    for i in range(num_steps):
        if method == "rk4":
            if stop_gradient:
                sol_t = _rk4_step_treemapped(f, (jax.lax.stop_gradient(p_traj[i]), jax.lax.stop_gradient(a_traj[i]), jax.lax.stop_gradient(window_traj[i])), t, h)
            else:
                sol_t = _rk4_step_treemapped(f, (p_traj[i], a_traj[i], window_traj[i]), t, h)
        elif method == "euler":
            if stop_gradient:
                sol_t = _euler_step_treemapped(f, (jax.lax.stop_gradient(p_traj[i]), jax.lax.stop_gradient(a_traj[i]), jax.lax.stop_gradient(window_traj[i])), t, h)
            else:
                sol_t = _euler_step_treemapped(f, (p_traj[i], a_traj[i], window_traj[i]), t, h)
        else:
            raise ValueError(f"Unknown method: {method}")

        p_traj = p_traj.at[i + 1].set(sol_t[0])
        a_traj = a_traj.at[i + 1].set(sol_t[1])
        window_traj = window_traj.at[i + 1].set(sol_t[2])
        t += h

    # Swap time and batch dims, so this func returns [batch, time, ...]
    p_traj_hat = jnp.transpose(p_traj, axes=(1, 0, 2, 3))
    a_traj_hat = jnp.transpose(a_traj, axes=(1, 0, 2, 3))
    window_traj_hat = jnp.transpose(window_traj, axes=(1, 0, 2, 3))

    return p_traj_hat, a_traj_hat, window_traj_hat


solve_ode = jit(
    vmap(_solve_ode, in_axes=(None, 0, None, None, None)), static_argnums=(0, 2, 3, 4)
)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # harmonic oscillator
    # def F(x, t):
    #     return jnp.array([x[1], -x[0]])
    
    # pendulum
    def F(x, t):
        return jnp.array([x[1], -jnp.sin(x[0])])


    # IC
    x0 = random.normal(random.PRNGKey(42), (100, 2))
    t0 = 0.0
    tf = 10.0
    h = 0.01

    trajectory = solve_ode(F, x0, t0, tf, h)

    for traj in trajectory:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.savefig("pendulum.png")

    
