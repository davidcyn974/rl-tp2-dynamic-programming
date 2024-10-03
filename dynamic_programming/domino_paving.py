# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    f_0 = 1 
    f_1 = 0
    g_0 = 0 
    g_1 = 1
    def f(n: int) -> int:
        if n == 0:
            return 1
        if n == 1:
            return 0
        return f(n - 2) + 2 * g(n - 1)

    def g(n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        return f(n - 1) + g(n - 2)

    if n % 2 == 1:
        return 0

    return f(n)
    # END SOLUTION
