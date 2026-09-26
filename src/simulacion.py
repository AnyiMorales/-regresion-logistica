"""
Simulación de datos bajo un modelo logístico generador conocido (Ejercicio 3).
"""

import numpy as np

from .modelo import p_theta


def generar_datos(n, theta_star, rng, x_min=-2.0, x_max=2.0):
    """
    Genera n observaciones (x_i, y_i) según:
        X_i  ~ Uniforme[x_min, x_max]
        Y_i | X_i ~ Bernoulli( p_theta_star(X_i) )

    Parámetros
    ----------
    n          : tamaño de muestra
    theta_star : (theta0*, theta1*), el modelo "verdadero" que genera los datos
    rng        : generador de numpy, p.ej. np.random.default_rng(2026)
                 (se pasa desde afuera para que la secuencia de números
                 aleatorios sea reproducible y compartida entre los n
                 considerados, tal como pide el enunciado)

    Devuelve
    --------
    x : np.ndarray de forma (n,)
    y : np.ndarray de forma (n,), valores en {0, 1}
    """
    x = rng.uniform(x_min, x_max, size=n)
    p = p_theta(theta_star, x)
    y = rng.binomial(n=1, p=p)  # Bernoulli(p) = Binomial(1, p)
    return x, y
