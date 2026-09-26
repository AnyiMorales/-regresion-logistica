"""
Ajuste del modelo logístico por minimización numérica del riesgo empírico.
"""

import numpy as np
from scipy.optimize import minimize

from .modelo import riesgo_empirico


def ajustar_modelo(x, y, theta_init=None, maxiter=None):
    """
    Encuentra theta_hat que minimiza R_S(theta) para los datos (x, y).

    Parámetros
    ----------
    x, y       : arreglos de datos
    theta_init : punto inicial (theta0, theta1); por defecto (0, 0)
    maxiter    : límite de iteraciones para el optimizador (None = default de scipy)

    Devuelve
    --------
    theta_hat : np.ndarray de forma (2,), el óptimo encontrado
    resultado : el objeto OptimizeResult completo de scipy (por si se
                necesita inspeccionar convergencia, número de iteraciones, etc.)
    """
    if theta_init is None:
        theta_init = np.array([0.0, 0.0])

    opciones = {} if maxiter is None else {"maxiter": maxiter}

    resultado = minimize(
        fun=riesgo_empirico,
        x0=theta_init,
        args=(x, y),
        method="BFGS",
        options=opciones,
    )
    return resultado.x, resultado
