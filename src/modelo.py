"""
Modelo logístico: sigmoide, probabilidades p_theta(x) y riesgo empírico.

Convención de theta: theta = (theta0, theta1), donde
    p_theta(x) = sigmoid(theta0 + theta1 * x)
"""

import numpy as np


def sigmoid(t):
    """
    Función logística sigmoide.

        sigmoid(t) = 1 / (1 + exp(-t))

    Funciona igual sobre un escalar o sobre un arreglo de numpy
    (numpy aplica exp elemento a elemento).
    """
    return 1.0 / (1.0 + np.exp(-t))


def p_theta(theta, x):
    """
    Probabilidad P(Y=1 | X=x) bajo el modelo logístico.

    Parámetros
    ----------
    theta : array-like de forma (2,) -> (theta0, theta1)
    x     : escalar o arreglo numpy de forma (n,)

    Devuelve
    --------
    Probabilidades, misma forma que x.
    """
    theta0, theta1 = theta
    return sigmoid(theta0 + theta1 * x)


def riesgo_empirico(theta, x, y, eps=1e-12):
    """
    Riesgo empírico logístico (negativo de la log-verosimilitud promedio):

        R_S(theta) = -(1/n) * sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ]

    donde p_i = p_theta(x_i).

    El parámetro eps evita log(0) por errores de redondeo cuando p_i
    queda numéricamente en 0.0 o 1.0 (recorta las probabilidades al
    intervalo [eps, 1-eps] antes de tomar logaritmo).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = p_theta(theta, x)
    p = np.clip(p, eps, 1 - eps)  # estabilidad numérica: evita log(0)
    n = x.shape[0]
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
