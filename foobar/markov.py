# Lint as: python3
"""TODO(ncapule): DO NOT SUBMIT without one-line documentation for solution.

TODO(ncapule): DO NOT SUBMIT without a detailed description of solution.
"""


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


def determinant(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    accumulator = 0
    first_row = matrix[0]
    for x, cell in enumerate(first_row):
        submatrix = submatrix_without_rowcol(matrix, 0, x)
        subdeterminant = determinant(submatrix)
        if x % 2 == 0:
            accumulator += cell*subdeterminant
        else:
            accumulator -= cell*subdeterminant

    return accumulator


def matrix_of_minors(matrix):
    new_matrix = []

    n = len(matrix)
    for y in range(n):
        new_row = []
        for x in range(n):
            new_row.append(determinant(submatrix_without_rowcol(matrix, y, x)))
        new_matrix.append(new_row)

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
    determinant_of_original = float(determinant(matrix))

    matrix = adjoint(matrix)

    n = len(matrix)
    for y in range(n):
        for x in range(n):
            matrix[y][x] = matrix[y][x] / determinant_of_original

    return matrix


def find_t_and_s(matrix):
    t = 0

    n = len(matrix)
    for y in range(n):
        has_value = False
        for x in range(n):
            if matrix[y][x] != 0:
                has_value = True
                break

        if has_value:
            t += 1

    return t, n - t


def solution(m):

    pass


if __name__ == '__main__':
    # mat = [
    #     [0, 2, 1, 0, 0],
    #     [0, 0, 0, 3, 4],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    # ]
    mat = [
        [0, 1, 0, 0, 0, 1],
        [4, 0, 0, 3, 2, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
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

    print(inverse(mat))
    print(find_t_and_s(mat))
