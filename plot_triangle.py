from run import *
from simulate import *


if __name__ == '__main__': 

    Z  = 100         # Population size
    N  = 4           # Group size
    b  = 1.           # Endowment (individual's money/funds/...)
    c  = 0.1         # Amount of money individuals contribute
    Mc = 0.3         # Minimum collective contribution
    M  = 3.          # OR Minimum number of cooperators
    r  = 0.2         # If minimum is not met: All group participants lose their endowment with probability r, else: individuals retain their endowments
    pi_t = 0.03
    pi_e = 0.3
    n_e = 1.
    alpha = 0.
    mu    = 1/Z
    beta = 5.
    enhancement_factor = 1.4

    transitory = 10**2      # num of steps before we start counting
    nb_generations = 10**4  # num of steps where we do count
    nb_runs = 10            # num of different runs we average over

    strategy_labels = ["Defector", "Executor", "Cooperator"]

    game = CRDWithExecutor(strategies=[Defector(c, b), Executor(c, b, pi_t, pi_e, alpha), Cooperator(c, b)],
                    initial_endowment=b,
                    population_size=Z,
                    group_size=N,
                    cost=c,
                    risk=r,
                    alpha=alpha,
                    cooperation_threshold=M,
                    enhancement_factor=enhancement_factor,
                    pi_t=pi_t,
                    pi_e=pi_e,
                    n_e=n_e,
                    mu=mu)

    payoffs = game.calculate_payoffs()

    fig, ax = plt.subplots(figsize=(15,10))

    simplex = egt.plotting.Simplex2D(discrete=True, size=Z, nb_points=Z+1)
    evolver = egt.analytical.StochDynamics(3, payoffs, Z, N, mu)
    calculate_gradients = lambda u: Z*evolver.full_gradient_selection(u, beta)
    
    #sd = evolver.calculate_stationary_distribution(beta)

    sd = estimate_stationary_distribution(
        game=game,
        nb_runs=nb_runs,
        transitory=transitory,
        nb_generations=nb_generations,
        beta=beta,
        mu=mu,
        Z=Z,
    )

    group_achievement = sum([
        sd[i]*game.aG(i) for i in range(len(sd))
    ])

    print(group_achievement)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * Z).astype(np.int64)
    result = np.asarray([[evolver.full_gradient_selection(v_int[:, i, j], beta) for j in range(v_int.shape[2])] for i in range(v_int.shape[1])]).swapaxes(0, 1).swapaxes(0, 2)
    
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)
    Ux = xy_results[:, :, 0].astype(np.float64)
    Uy = xy_results[:, :, 1].astype(np.float64)

    plot = (simplex.add_axis(ax=ax)
        .apply_simplex_boundaries_to_gradients(Ux, Uy)
        .draw_gradients(zorder=5)
        .add_colorbar()
        .add_vertex_labels(strategy_labels)
        .draw_stationary_distribution(sd, alpha=1, edgecolors='gray', cmap='binary',shading='gouraud', zorder=0)
    )

    ax.axis('off')
    ax.set_aspect('equal')
    plt.xlim((-.05,1.05))
    plt.ylim((-.02, simplex.top_corner + 0.05))
    plt.show()