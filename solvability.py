import numpy as np

def get_zero_tile_distance(puzzle_state):
    """
    Calculate the distance of the zero (empty) tile to its target position.
    Parameters: puzzle_state (List[List[int]]): The current state of the puzzle.
    Returns: int: The Manhattan distance from the current position of the zero tile to its target position.
    """
    dimension = len(puzzle_state)
    for row in range(dimension):
        for col in range(dimension):
            if puzzle_state[row][col] == 0:
                zero_row = row
                zero_col = col
                break
    if dimension % 2 != 0:
        target_row = dimension // 2
        target_col = dimension // 2
    else:
        target_row = dimension // 2
        target_col = dimension // 2 - 1
    distance = abs(target_row - zero_row) + abs(target_col - zero_col)
    return distance


def count_inversions(puzzle_state, model_state):
    """
    Count the number of inversions in the current puzzle state.
    Parameters: puzzle_state (List[List[int]]): The current state of the puzzle.
                model_state (List[List[int]]): The model state of the puzzle.
    Returns: int: The number of inversions.
    """
    puzzle_flat = [val for sublist in puzzle_state for val in sublist]
    model_flat = [val for sublist in model_state for val in sublist]
    inversions = 0
    for i in range(len(puzzle_flat)):
        for j in range(i, len(puzzle_flat)):
            if model_flat.index(puzzle_flat[i]) > model_flat.index(puzzle_flat[j]):
                inversions += 1
    return inversions


def is_puzzle_solvable(puzzle_state):
    """
    Check if the puzzle is solvable.
    Parameters: puzzle_state (List[List[int]]): The current state of the puzzle.
    Returns: bool: True if the puzzle is solvable, False otherwise.
    """
    dimension = len(puzzle_state)
    model_state = np.zeros((dimension, dimension), dtype=int)
    count, col, row, iteration = 1, 0, 0, 0

    while count < dimension * dimension:
        while col < dimension - iteration:
            model_state[row][col] = count
            count += 1
            col += 1
        col -= 1
        row += 1

        while row < dimension - iteration:
            model_state[row][col] = count
            count += 1
            row += 1
        row -= 1
        col -= 1

        while col >= iteration:
            model_state[row][col] = count
            count += 1
            col -= 1
        col += 1
        row -= 1

        while row > iteration:
            model_state[row][col] = count
            count += 1
            row -= 1
        row += 1
        col += 1
        iteration += 1

    if dimension % 2 == 0:
        model_state[dimension // 2][dimension // 2 - 1] = 0
    else:
        model_state[dimension // 2][dimension // 2] = 0

    zero_tile_distance = get_zero_tile_distance(puzzle_state.tolist())
    inversion_count = count_inversions(puzzle_state.tolist(), model_state.tolist())
    solvable = zero_tile_distance % 2 == inversion_count % 2
    return solvable
