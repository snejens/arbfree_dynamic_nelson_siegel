from .nss import C, lambda_, lambda_svensson

import sympy
from sympy import cse, ccode, numbered_symbols


tau, sigma_00, sigma_10, sigma_11, sigma_20, sigma_21, sigma_22 = sympy.var(
    'tau sigma_00 sigma_10 sigma_11 sigma_20 sigma_21 sigma_22', real=True)

Sigma = sympy.Matrix([[sigma_00, 0, 0],
                      [sigma_10, sigma_11, 0],
                      [sigma_20, sigma_21, sigma_22]])

print("3\n---------------------\n\n")
replacements, reduced_exprs = cse(C(0, tau, None, Sigma, lambda_, lambda_svensson), numbered_symbols("t"))

for symbol, subexpr in replacements:
    print(f"const double {symbol} = {ccode(subexpr)};")

print(f"return {ccode(reduced_exprs[0])};")



sigma_30, sigma_31, sigma_32, sigma_33, sigma_40, sigma_41, sigma_42, sigma_43, sigma_44 = sympy.var(
    'sigma_30 sigma_31 sigma_32 sigma_33 sigma_40 sigma_41 sigma_42 sigma_43 sigma_44', real=True)
Sigma = sympy.Matrix([[sigma_00, 0, 0, 0, 0],
                      [sigma_10, sigma_11, 0, 0, 0],
                      [sigma_20, sigma_21, sigma_22, 0, 0],
                      [sigma_30, sigma_31, sigma_32, sigma_33, 0],
                      [sigma_40, sigma_41, sigma_42, sigma_43, sigma_44]])

print("5\n---------------------\n\n")
replacements, reduced_exprs = cse(C(0, tau, None, Sigma, lambda_, lambda_svensson), numbered_symbols("t"))

for symbol, subexpr in replacements:
    print(f"const double {symbol} = {ccode(subexpr)};")

print(f"return {ccode(reduced_exprs[0])};")
