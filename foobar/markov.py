# Lint as: python3
"""TODO(ncapule): DO NOT SUBMIT without one-line documentation for solution.

TODO(ncapule): DO NOT SUBMIT without a detailed description of solution.
"""

from fractions import gcd


class Fraction:
    def __init__(self, upper, lower=1):
        self.upper = upper
        self.lower = lower

    def __mul__(self, other):
        return Fraction(self.upper * other.upper, self.lower * other.lower)

    def __div__(self, other):
        upper = self.upper * other.lower
        lower = self.lower * other.upper
        if lower < 0:
            upper = upper * -1
            lower = abs(lower)

        return Fraction(upper, lower)

    def __neg__(self):
        return Fraction(-self.upper, self.lower)

    def __sub__(self, other):
        denum = gcd(self.lower, other.lower)
        if denum == 0:
            return Fraction(0)
        lcm = abs(self.lower * other.lower) / denum
        if lcm == 0:
            return Fraction(0)

        self_upper = self.upper * lcm / self.lower
        other_upper = other.upper * lcm / other.lower
        upper = self_upper - other_upper

        return Fraction(upper, lcm)

    def __add__(self, other):
        denum = gcd(self.lower, other.lower)
        if denum == 0:
            return Fraction(0)
        lcm = abs(self.lower * other.lower) / denum
        if lcm == 0:
            return Fraction(0)

        self_upper = self.upper * lcm / self.lower
        other_upper = other.upper * lcm / other.lower
        upper = self_upper + other_upper

        return Fraction(upper, lcm)

    def __repr__(self):
        return "({}/{})".format(self.upper, self.lower)

    def simplify(self):
        denum = gcd(self.lower, self.upper)
        while denum > 1:
            self.lower /= denum
            self.upper /= denum
            denum = gcd(self.lower, self.upper)
        return self


def lcm(xs):
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return abs(xs[0] * xs[1]) / gcd(xs[0], xs[1])
    return lcm([xs[0], lcm(xs[1:])])


def normalize_fractions(fractions):
    fractions = [v.simplify() for v in fractions]
    all_lowers = [v.lower for v in fractions]
    all_lcm = lcm(all_lowers)
    for i in range(len(fractions)):
        fractions[i].upper = fractions[i].upper * all_lcm / fractions[i].lower
        fractions[i].lower = all_lcm
    return fractions


def submatrix_without_rowcol(matrix, no_y, no_x):
    new_matrix = []

    for y, row in enumerate(matrix):
        if y == no_y:
            continue
        new_row = []
        for x, cell in enumerate(row):
            if x == no_x:
                continue
            new_row.append(cell)
        new_matrix.append(new_row)

    return new_matrix


def submatrix_by_range(matrix, sy, sx, ey, ex):
    new_matrix = []

    for y in range(sy, ey):
        new_row = []
        for x in range(sx, ex):
            new_row.append(matrix[y][x])
        new_matrix.append(new_row)

    return new_matrix


def is_zero_matrix(matrix):
    n = len(matrix)
    for y in range(n):
        for x in range(n):
            if matrix[y][x].upper != 0:
                return False
    return True


def determinant(matrix):
    if len(matrix) == 2:
        # Operator!
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    # The determinant of a zero matrix is 1!
    if is_zero_matrix(matrix):
        return Fraction(1)

    accumulator = Fraction(0)
    first_row = matrix[0]
    for x, cell in enumerate(first_row):
        submatrix = submatrix_without_rowcol(matrix, 0, x)
        subdeterminant = determinant(submatrix)
        if x % 2 == 0:
            # Operator!
            accumulator = accumulator + (cell*subdeterminant)
        else:
            # Operator!
            accumulator = accumulator - (cell*subdeterminant)

    return accumulator


def matrix_of_minors(matrix):
    new_matrix = []

    n = len(matrix)
    for y in range(n):
        new_row = []
        for x in range(n):
            new_row.append(determinant(submatrix_without_rowcol(matrix, y, x)))
        new_matrix.append(normalize_fractions(new_row))

    return new_matrix


def matrix_of_cofactors(matrix):
    matrix = matrix_of_minors(matrix)

    n = len(matrix)
    for y in range(n):
        for x in range(n):
            if (y + x) % 2 > 0:
                matrix[y][x] = -matrix[y][x]

    return matrix


def adjoint(matrix):
    matrix = matrix_of_cofactors(matrix)

    n = len(matrix)
    for y in range(n):
        for x in range(y + 1, n):
            tmp = matrix[y][x]
            matrix[y][x] = matrix[x][y]
            matrix[x][y] = tmp

    return matrix


def inverse(matrix):
    '''
    Retrieves the inverse of a matrix.

    Reference: https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html
    '''
    determinant_of_original = determinant(matrix)

    matrix = adjoint(matrix)

    n = len(matrix)
    for y in range(n):
        for x in range(n):
            # Operator!
            matrix[y][x] = matrix[y][x] / determinant_of_original

    return matrix


def find_t_and_s(matrix):
    t = 0

    n = len(matrix)
    for y in range(n):
        has_value = False
        for x in range(n):
            if matrix[y][x].upper != 0:
                has_value = True
                break

        if has_value:
            t += 1

    return t, n - t


def identity_minus_matrix(matrix):
    n = len(matrix)
    for y in range(n):
        for x in range(n):
            if x == y:
                # Operator!
                matrix[y][x] = Fraction(1) - matrix[y][x]
            else:
                # Operator!
                matrix[y][x] = -matrix[y][x]

    return matrix


def dot_product(matrix_a, matrix_b):
    N = len(matrix_b[0])

    new_matrix = []
    for row_a in matrix_a:
        new_row = []
        for x in range(N):
            new_cell = Fraction(0)
            for c, cell_a in enumerate(row_a):
                # Operator!
                new_cell = new_cell + (cell_a * matrix_b[c][x])
            new_row.append(new_cell)
        new_matrix.append(normalize_fractions(new_row))

    return new_matrix


def convert_to_fractions(matrix):
    new_matrix = []
    for row in matrix:
        row_sum = 0
        for cell in row:
            row_sum = row_sum + cell

        new_row = []
        for cell in row:
            new_row.append(Fraction(cell, row_sum))
        new_matrix.append(new_row)
    return new_matrix


def move_terminal_states_at_end(matrix):
    # Append index at the start of each row
    def rank(row):
        return sum([x for x in row[1:]]) == 0

    for _ in range(len(matrix)):
        for j in range(len(matrix)-1):
            if rank(matrix[j]) > rank(matrix[j+1]):
                left = matrix[j]
                right = matrix[j+1]
                matrix[j] = right
                matrix[j+1] = left

                for k in range(len(matrix)):
                    tmp = matrix[k][j]
                    matrix[k][j] = matrix[k][j+1]
                    matrix[k][j+1] = tmp

    return matrix


def solution(mat):
    '''
    Reference: https://brilliant.org/wiki/absorbing-markov-chains/
    '''

    # Edge case, when matrix is a 1x1 matrix.
    if len(mat) == 1:
        return [1, 1]

    # Sort matrix to move terminal states at the end.
    mat = move_terminal_states_at_end(mat)

    # Convert all elements of the matrix to fraction.
    mat = convert_to_fractions(mat)

    # According to the reference, we need to find t and s.
    t, s = find_t_and_s(mat)

    Q = submatrix_by_range(mat, 0, 0, t, t)

    # Fundamental matrix is (I - Q)^-1
    fundamental_matrix = inverse(identity_minus_matrix(Q))

    R = submatrix_by_range(mat, 0, t, t, s+t)

    # Normalize the resulting fractions.
    normalized = normalize_fractions(dot_product(fundamental_matrix, R)[0])

    # Return to a format that the foobar problem wants anyway.
    return [v.upper for v in normalized] + [normalized[0].lower]


if __name__ == '__main__':
    mat = [
        [0, 1, 0, 0, 0, 1],
        [4, 0, 0, 3, 2, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    mat = [0]
    # mat = [
    #     [0, 2, 1, 0, 0],
    #     [0, 0, 0, 3, 4],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    # ]
    # mat = [
    #     [0, 1, 0, 0, 2],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 3, 4, 0],
    # ]
    # mat = [
    #     [1, 3, 2],
    #     [1, 3, 123],
    #     [0, 0, 0],
    # ]
    # mat = [
    #     [0, 1, 0, 1, 0],
    #     [1, 0, 1, 0, 0],
    #     [0, 1, 0, 0, 1],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    # ]
    # mat = [
    #     [0, 1, 2, 3, 4],
    #     [5, 6, 7, 8, 9],
    #     [10, 11, 12, 13, 14],
    #     [15, 16, 17, 18, 19],
    #     [20, 21, 22, 23, 24],
    # ]
    # mat = [
    #     [1, 3, -2, 1],
    #     [5, 1, 0, -1],
    #     [0, 1, 0, -2],
    #     [2, -1, 0, 3],
    # ]
    # mat = [
    #     [6, 1, 1],
    #     [4, -2, 5],
    #     [2, 8, 7],
    # ]
    # mat = [
    #     [1, 2, 3],
    #     [0, -4, 1],
    #     [0, 3, -1],
    # ]
    # mat = [
    #     [3, 0, 2],
    #     [2, 0, -2],
    #     [0, 1, 1],
    # ]
    # mat = [
    #     [4, 6],
    #     [3, 8],
    # ]
    # mat = [[10, 11, 12], [15, 16, 17], [20, 21, 22]]

    # print(inverse(mat))

    print(solution(mat))
