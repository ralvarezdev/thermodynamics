import numpy as np
from scipy.optimize import fsolve

# Constante de los gases en J/(mol·K)
R = 8.314


def calculate_a(Tc, Pc):
    return 0.45724 * (R**2) * (Tc**2) / Pc


def calculate_b(Tc, Pc):
    return 0.07780 * R * Tc / Pc


def calculate_alpha(T, Tc, omega):
    Tr = T / Tc
    m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    return (1 + m * (1 - np.sqrt(Tr))) ** 2


def calculate_A_mix(a_mix, alpha_mix, P, T):
    return a_mix * alpha_mix * P / (R * T) ** 2


def calculate_B_mix(b_mix, P, T):
    return b_mix * P / (R * T)


def calculate_Z(A_mix, B_mix):
    # Coefficients of the cubic Peng-Robinson equation
    coeffs = [
        1,
        -(1 - B_mix),
        A_mix - 2 * B_mix - 3 * B_mix**2,
        -(A_mix * B_mix - B_mix**2 - B_mix**3),
    ]
    # Solving the cubic equation for Z
    Z_roots = np.roots(coeffs)
    # Considering the real roots only
    Z_roots = Z_roots[np.isreal(Z_roots)]
    # Taking the maximum real root (Z must be positive and largest root is typically used)
    Z1 = min(Z_roots)
    Z2 = max(Z_roots)
    print("Z1(Líquido saturado) es", Z1)
    print("Z2(Vapor saturado) es", Z2)

    return


def peng_robinson_Z_mixture(components, fractions, Tc, Pc, omega, T, P):
    a = np.array([calculate_a(Tc[i], Pc[i]) for i in range(len(components))])
    b = np.array([calculate_b(Tc[i], Pc[i]) for i in range(len(components))])
    alpha = np.array(
        [calculate_alpha(T, Tc[i], omega[i]) for i in range(len(components))]
    )

    # Calculate the mixed parameters a_mix and b_mix
    a_mix = 0
    for i in range(len(components)):
        for j in range(len(components)):
            a_mix += fractions[i] * fractions[j] * np.sqrt(a[i] * a[j])

    b_mix = np.dot(fractions, b)
    alpha_mix = np.dot(fractions, alpha)

    # Calculate A_mix and B_mix
    A_mix = calculate_A_mix(a_mix, alpha_mix, P, T)
    B_mix = calculate_B_mix(b_mix, P, T)

    # Calculate Z
    return calculate_Z(A_mix, B_mix)


# Ejemplo de uso
components = ["metano", "etano", "propano"]
fractions = [0.2, 0.3, 0.7]
Tc = [190.6, 305.3, 369.8]  # Temperaturas críticas en K
Pc = [4590000, 4872000, 4248000]  # Presiones críticas en Pa
omega = [0.012, 0.100, 0.152]  # Factores acéntricos
T = 303.15  # Temperatura del sistema en K
P = 400000  # Presión total en Pa

Z = peng_robinson_Z_mixture(components, fractions, Tc, Pc, omega, T, P)
