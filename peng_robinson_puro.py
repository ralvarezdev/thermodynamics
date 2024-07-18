import numpy as np

# Constante de los gases en J/(mol·K)
R = 8.314

def calculate_a(Tc, Pc):
    Pc = Pc * 100000  # Convertir presión crítica de bar a Pa
    return 0.45724 * (R ** 2) * (Tc ** 2) / Pc

def calculate_b(Tc, Pc):
    Pc = Pc * 100000  # Convertir presión crítica de bar a Pa
    return 0.07780 * R * Tc / Pc

def calculate_alpha(T, Tc, omega):
    Tr = T / Tc
    m = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    return (1 + m * (1 - np.sqrt(Tr))) ** 2

def calculate_A(a, alpha, P, T):
    P = P * 100000  # Convertir presión del sistema de bar a Pa
    return a * alpha * P / (R * T) ** 2

def calculate_B(b, P, T):
    P = P * 100000  # Convertir presión del sistema de bar a Pa
    return b * P / (R * T)

def calculate_Z_roots(A, B):
    # Coefficients of the cubic Peng-Robinson equation
    coeffs = [1, -(1 - B), A - 2 * B - 3 * B ** 2, -(A * B - B ** 2 - B ** 3)]
    # Solving the cubic equation for Z
    Z_roots = np.roots(coeffs)
    # Considering the real roots only
    Z_roots = Z_roots[np.isreal(Z_roots)]
    # Sorting the real roots
    Z_roots = np.sort(Z_roots.real)
    return Z_roots

def peng_robinson_Z_pure(Tc, Pc, omega, T, P):
    a = calculate_a(Tc, Pc)
    b = calculate_b(Tc, Pc)
    alpha = calculate_alpha(T, Tc, omega)
    
    A = calculate_A(a, alpha, P, T)
    B = calculate_B(b, P, T)
    
    Z_roots = calculate_Z_roots(A, B)
    
    if len(Z_roots) == 3:
        Z_liquid = Z_roots[0]  # Menor raíz
        Z_vapor = Z_roots[2]   # Mayor raíz
    else:
        # En algunos casos puede haber una o dos raíces reales
        Z_liquid = Z_roots[0]
        Z_vapor = Z_roots[-1]
    
    return Z_liquid, Z_vapor

# Ejemplo de uso
Tc = 417.2  # Temperatura crítica en K
Pc = 77.1  # Presión crítica en bar
omega = 0.069  # Factor acéntrico
T = 343.15  # Temperatura del sistema en K
P = 22.49  # Presión total en bar

Z_liquid, Z_vapor = peng_robinson_Z_pure(Tc, Pc, omega, T, P)
print(f'Factor de compresibilidad Z (líquido saturado): {Z_liquid}')
print(f'Factor de compresibilidad Z (vapor saturado): {Z_vapor}')
