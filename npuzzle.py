import sys
import os
import argparse
import random
import copy
import heapq
import matplotlib.pyplot as plt
import numpy as np

from time import time
from typing import List, Set, TextIO

from solvability import is_puzzle_solvable

PAUSE_TIME = 0.1  # global variable for pause time between moves in graphical mode
graphical = False

class Node:
    """
    Class representing a node in the puzzle.
    """
    def __init__(self, tiles: List[int] = None, parent=None, g=0, h=0, f=0):
        """
        Initialize the node.
        Args: tiles (List[int], optional): The tiles of the puzzle.
              parent (Node, optional): The parent of the node.
              g (int, optional): The cost to reach this node.
              h (int, optional): The heuristic cost estimate of the cheapest path from this node to the goal.
              f (int, optional): The estimated cost of the cheapest solution through this node.
        """
        if tiles is None:
            tiles = list()
        self.tiles = tiles
        self.g = g
        self.h = h
        self.f = f
        self.parent = parent

    def is_goal(self):
        """
        Check if this node is the goal.
        Returns: bool: True if this node is the goal, False otherwise.
        """
        return self.tiles == goal.tiles

    def get_empty(self):
        """
        Get the position of the empty tile.
        Returns: tuple: The position of the empty tile.
        """
        tiles = self.tiles
        for i in range(size * size):
            if tiles[i] == 0:
                return i // size, i % size
        return 0, 0

    def manhattan_distance(self):
        """
        Calculate the Manhattan distance from the current state to the goal state.
        This is the sum of the distances (in the horizontal and vertical directions)
        from each tile to its goal position. It represents the minimum number of moves
        that each tile must make to reach its goal position.
        """
        total_sum = 0
        tiles = self.tiles

        for i in range(size * size):
            if tiles[i] == 0:
                continue
            else:
                x, y = goal_positions[tiles[i]]
                total_sum += abs(x - i // size) + abs(y - i % size)

        self.h = total_sum
        self.g = self.parent.g + 1
        self.f = self.h
        if algorithm != 'greedy':
            self.f += self.g

    def euclidean_distance(self):
        """
        Calculate the Euclidean distance from the current state to the goal state.
        The Euclidean distance is also knowed as "as the crow flies" distance.
        """
        total_sum = 0
        tiles = self.tiles

        for i in range(size * size):
            if tiles[i] == 0:
                continue
            else:
                goal_x, goal_y = goal_positions[tiles[i]]
                current_x, current_y = i // size, i % size
                total_sum += ((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2) ** 0.5

        self.h = total_sum
        self.g = self.parent.g + 1
        self.f = self.h
        if algorithm != 'greedy':
            self.f += self.g

    def hamming_distance(self):
        """
        Calculate the Hamming distance from the current state to the goal state.
        Misplaced Tiles (Hamming distance): This is the total number of tiles
        that are not in their correct place.
        """
        total_sum = 0
        tiles = self.tiles

        for i in range(size * size):
            if tiles[i] == 0:
                continue
            elif tiles[i] != goal.tiles[i]:
                total_sum += 1

        self.h = total_sum
        self.g = self.parent.g + 1
        self.f = self.h
        if algorithm != 'greedy':
            self.f += self.g

    def linear_conflict_manhattan(self):
        """
        Calculate the linear conflict from the current state to the goal state.
        This involves counting the number of pairs of tiles that are in the wrong
        order compared to the goal. Use manhattan, then add 2 for each pair of
        tiles that are in the same row or column and must be swapped.
        """
        self.manhattan_distance()
        total_sum = self.h
        tiles = self.tiles

        for row in range(size):
            row_tiles = [tiles[i] for i in range(row * size, (row + 1) * size) if tiles[i] != 0]
            for i in range(len(row_tiles)):
                for j in range(i + 1, len(row_tiles)):
                    if (goal_positions[row_tiles[i]][0] == goal_positions[row_tiles[j]][0]
                            and goal_positions[row_tiles[i]][1] > goal_positions[row_tiles[j]][1]):
                        total_sum += 2

        for col in range(size):
            col_tiles = [tiles[i * size + col] for i in range(size) if tiles[i * size + col] != 0]
            for i in range(len(col_tiles)):
                for j in range(i + 1, len(col_tiles)):
                    if goal_positions[col_tiles[i]][1] == goal_positions[col_tiles[j]][1] and \
                            goal_positions[col_tiles[i]][0] > goal_positions[col_tiles[j]][0]:
                        total_sum += 2

        self.h = total_sum
        self.g = self.parent.g + 1
        self.f = self.h
        if algorithm != 'greedy':
            self.f += self.g

    def __eq__(self, other):
        """
        Check if this node is equal to another node.
        Args: other (Node): The other node.
        Returns: bool: True if they are equal, False otherwise.
        """
        if other is None:
            return False
        return self.tiles == other.tiles and self.f == other.f

    def __lt__(self, other):
        """
        Check if this node is less than another node.
        Args: other (Node): The other node.
        Returns: bool: True if this node is less than the other node, False otherwise.
        """
        return self.f < other.f

    def __hash__(self):
        """
        Calculate the hash of this node.
        Returns: int: The hash of this node.
        """
        return hash(tuple(self.tiles))

    def get_children(self):
        """
        Generate the children of this node.
        Returns: list: The children of this node.
        """
        tiles = self.tiles
        x, y = self.get_empty()
        new_tiles = []

        if (x + 1) < size:
            new = copy.deepcopy(tiles)
            new[x * size + y] = new[(x + 1) * size + y]
            new[(x + 1) * size + y] = 0
            new_tiles.append(new)
        if (x - 1) > -1:
            new = copy.deepcopy(tiles)
            new[x * size + y] = new[(x - 1) * size + y]
            new[(x - 1) * size + y] = 0
            new_tiles.append(new)
        if (y + 1) < size:
            new = copy.deepcopy(tiles)
            new[x * size + y] = new[x * size + y + 1]
            new[x * size + y + 1] = 0
            new_tiles.append(new)
        if (y - 1) > -1:
            new = copy.deepcopy(tiles)
            new[x * size + y] = new[x * size + y - 1]
            new[x * size + y - 1] = 0
            new_tiles.append(new)

        ret = []
        for i in new_tiles:
            child = Node(i, self)
            if heuristic == "manhattan":
                child.manhattan_distance()
            elif heuristic == "euclidean":
                child.euclidean_distance()
            elif heuristic == "hamming":
                child.hamming_distance()
            elif heuristic == "lconflict":
                child.linear_conflict_manhattan()

            if self.parent is not None and child.tiles == self.parent.tiles:
                continue

            ret.append(child)

        return ret


class PriorityQueue:
    """
    Class representing a priority queue.
    """
    def __init__(self):
        """
        Initialize the priority queue.
        """
        self.elements = []

    def __len__(self):
        """
        Get the number of elements in the priority queue.
        Returns: int: The number of elements.
        """
        return len(self.elements)

    def __getitem__(self, item):
        """
        Get the item at the given index.
        Args: item (int): The index of the item.
        Returns: Any: The item at the given index.
        """
        return self.elements[item]

    def empty(self):
        """
        Check if the priority queue is empty.
        Returns: bool: True if the priority queue is empty, False otherwise.
        """
        return len(self.elements) == 0

    def put(self, node: Node):
        """
        Add a node to the priority queue.
        Args: node (Node): The node to add.
        """
        heapq.heappush(self.elements, node)

    def get(self):
        """
        Remove and return the smallest node from the priority queue.
        Returns: Node: The smallest node.
        """
        return heapq.heappop(self.elements)


goal: Node = Node()
size: int = 0
goal_positions: dict = {}
heuristic: str = "manhattan"
randomize: int = 100
algorithm: str = "astar"


def generate_random_puzzle(puzzle_size: int) -> Node:
    """
    Generates a random valid puzzle of the specified size.

    This function takes the size of the puzzle as input and generates a random valid puzzle
    of that size. The puzzle is generated by starting with an ordered spiral puzzle and then
    performing a random sequence of valid moves, moving the zero tile with an adjacent tile
    (vertical or horizontal), [randomize] times.

    Parameters: puzzle_size (int): The size of the puzzle.

    Returns: Node: The initial state of the generated puzzle.
    """
    tiles = generate_ordered_spiral_puzzle(puzzle_size)
    initial_state = Node(tiles)

    for _ in range(randomize):
        empty_x, empty_y = initial_state.get_empty()

        possible_moves = []
        if empty_x > 0:
            possible_moves.append((empty_x - 1, empty_y))
        if empty_x < puzzle_size - 1:
            possible_moves.append((empty_x + 1, empty_y))
        if empty_y > 0:
            possible_moves.append((empty_x, empty_y - 1))
        if empty_y < puzzle_size - 1:
            possible_moves.append((empty_x, empty_y + 1))

        move_x, move_y = random.choice(possible_moves)
        empty_pos = empty_x * puzzle_size + empty_y
        move_pos = move_x * puzzle_size + move_y

        initial_state.tiles[empty_pos], initial_state.tiles[move_pos] = \
            initial_state.tiles[move_pos], initial_state.tiles[empty_pos]

    return initial_state


def generate_ordered_spiral_puzzle(puzzle_size: int) -> list:
    """
    Generates an ordered spiral puzzle of the specified size.
    This function takes the size of the puzzle as input and generates an ordered spiral puzzle
    of that size. The puzzle is generated by starting with an empty grid and filling it in a
    spiral pattern with the tile values.
    Parameters: puzzle_size (int): The size of the puzzle.
    Returns: list: The ordered spiral puzzle tiles.
    """
    tiles = [[0] * puzzle_size for _ in range(puzzle_size)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_idx = 0
    current_x, current_y = 0, 0
    count = 1

    available_numbers = list(range(1, puzzle_size * puzzle_size + 1))

    while count <= puzzle_size * puzzle_size:
        tiles[current_x][current_y] = available_numbers[count - 1]
        count += 1

        next_x = current_x + directions[direction_idx][0]
        next_y = current_y + directions[direction_idx][1]

        if (
                next_x < 0
                or next_x >= puzzle_size
                or next_y < 0
                or next_y >= puzzle_size
                or tiles[next_x][next_y] != 0
        ):
            direction_idx = (direction_idx + 1) % 4

        current_x += directions[direction_idx][0]
        current_y += directions[direction_idx][1]

    tiles = [tile for row in tiles for tile in row]
    tiles[tiles.index(max(tiles))] = 0

    return tiles


def generate_goal():
    """
    Generates the goal state for the puzzle.
    """
    global goal, goal_positions

    tiles = np.arange(size * size)

    count: int = 1

    col: int = 0
    row: int = 0
    iteration: int = 0

    while count < size * size:

        while col < size - iteration:
            tiles[row * size + col] = count
            goal_positions[count] = (row, col)
            count += 1
            col += 1
        if count > size * size:
            break
        col -= 1
        row += 1

        while row < size - iteration:
            tiles[row * size + col] = count
            goal_positions[count] = (row, col)
            count += 1
            row += 1
        if count > size * size:
            break
        row -= 1
        col -= 1

        while col >= iteration:
            tiles[row * size + col] = count
            goal_positions[count] = (row, col)
            count += 1
            col -= 1
        if count > size * size:
            break
        col += 1
        row -= 1

        while row > iteration:
            tiles[row * size + col] = count
            goal_positions[count] = (row, col)
            count += 1
            row -= 1
        if count > size * size:
            break
        row += 1
        col += 1
        iteration += 1

    if size % 2 == 0:
        tiles[size // 2 * size + size // 2 - 1] = 0
        goal_positions[0] = (size // 2, size // 2 - 1)
    else:
        tiles[size // 2 * size + size // 2] = 0
        goal_positions[0] = (size // 2, size // 2)

    goal.tiles = list(tiles)

    print("Goal state:")
    printNodeText(goal)


def printNode(node: Node):
    """
    Visualizes the state of the puzzle using matplotlib.
    The function first clears the current figure (if it exists), then reshapes the
    flat list of tiles into a 2D array to represent the puzzle grid.
    It then uses matplotlib to create an image of the grid, where each tile's value
    is represented as a different color. The value of each tile is also printed at
    the center of the corresponding square in the grid.
    After drawing the grid, the function pauses for a set amount of time (determined
    by the global variable PAUSE_TIME) before returning.
    Parameters: node (Node): The current state of the puzzle.
    Returns: None
    """
    plt.clf()
    tiles = node.tiles
    grid = np.array(tiles).reshape((size, size))

    plt.imshow(grid, cmap='viridis')
    for i in range(size):
        for j in range(size):
            plt.text(j, i, grid[i, j],
                     ha="center", va="center", color="black")
    plt.xticks([])
    plt.yticks([])
    plt.title("Solution")
    plt.draw()
    plt.pause(PAUSE_TIME)


def printNodeText(node: Node, file: TextIO = sys.stdout):
    """
    Prints the current state of the puzzle in the console.
    This function takes the state of the puzzle and prints it as a 2D grid in
    the console. Each row of the grid is printed on a new line, and within a row,
    tile values are separated by a space.
    Parameters: node (Node): The current state of the puzzle.
                file (TextIO): The file to write the output to. If None, prints to console.
    Returns: None
    """
    tiles = node.tiles
    for i in range(size):
        for j in range(size):
            if file is not None:
                print(tiles[i * size + j], end=" ", file=file)
        if file is not None:
            print(file=file)
    if file is not None:
        print(file=file)


def parseArgs():
    """
    This function parses the command line arguments and reads the puzzle file or generates a random puzzle.
    The file is expected to contain the size of the puzzle on the first line and
    the initial state of the puzzle on the subsequent lines. The function also
    accepts optional arguments to specify the algorithm, the heuristic function and whether to
    display the solution graphically.
    The function performs error checking for:
    - Presence of file and readability
    - Puzzle size being more than 2
    - Correct number of tiles in each row
    - Presence of non-numeric values
    - Duplicate tiles
    - Absence of an empty tile (0)
    The heuristic function is selected based on the argument, with "manhattan"
    being the default. The options are "manhattan", "euclidean", and "hamming".
    The search algorithm is selected based on the argument, with "astar" being
    the default. The options are "astar" and "greedy".
    If the graphical flag is set, the puzzle and its solution will be displayed
    graphically (maximum size 25).
    Parameters: None
    Returns: A tuple containing:
             - The initial state of the puzzle (Node object)
             - A boolean indicating if the solution should be displayed graphically.
    Raises: SystemExit: If an error occurs during file reading, validation, or
                        incorrect usage of arguments.
    """
    global size, heuristic, algorithm

    parser = argparse.ArgumentParser(description="Solve the n-puzzle.")
    parser.add_argument("file", nargs="?", help="The puzzle file to solve, mandatory without -R option.")
    parser.add_argument("-G", action="store_true", help="Display solution graphically (size 25 max).")
    parser.add_argument("-H", choices=['manhattan', 'lconflict', 'euclidean', 'hamming'],
                        default='manhattan', help="Choose the heuristic function (default: manhattan)")
    parser.add_argument("-R", type=int, metavar="[3-100]",
                        help="Generate a random puzzle of the specified size (3-100).")
    parser.add_argument("-A", choices=['astar', 'greedy'], default='astar',
                        help="Choose the search algorithm (default: astar)")
    args = parser.parse_args()

    file = args.file
    graphical = args.G
    heuristic = args.H
    algorithm = args.A

    if file is not None and args.R is not None:
        print("Error: Cannot specify both a file and the random option.")
        sys.exit(1)

    if file is not None:
        if not os.path.isfile(file) or not os.access(file, os.R_OK):
            print("Error opening file.")
            sys.exit(1)

        with open(file) as f:
            lines = [line for line in f.readlines() if not line.startswith("#")]

        try:
            size = int(lines[0].strip())
            if size <= 2:
                print("Puzzle size should be more than 2.")
                sys.exit(1)
            lines = lines[1:]
        except (ValueError, IndexError):
            print("Invalid or missing puzzle size.")
            sys.exit(1)

        tiles = np.zeros((size, size), dtype=int)
        tile_set = set()
        if size > 25 and graphical:
            print("Puzzle size too large (25 max) for graphical display. Graphical display will be disabled.")
            graphical = False

        for i in range(size):
            row = lines[i].split()
            if len(row) != size:
                print(f"Invalid number of tiles in row {i + 1}. Expected {size} but got {len(row)}.")
                sys.exit(1)
            for j in range(size):
                try:
                    tile = int(row[j])
                    if tile in tile_set:
                        print("Duplicate tile found: " + str(tile))
                        sys.exit(1)
                    tile_set.add(tile)
                    tiles[i][j] = tile
                except ValueError:
                    print(f"Non-numeric value '{row[j]}' found at row {i + 1}, column {j + 1}")
                    sys.exit(1)

        if 0 not in tile_set:
            print("No empty tile (0) found in the puzzle.")
            sys.exit(1)

        starting: Node = Node()
        starting.tiles = tiles.flatten().tolist()

        if graphical:
            print("Starting state (graphical):")
            printNode(starting)
        else:
            print("Starting state:")
            printNodeText(starting)

        return starting, graphical

    elif args.R is not None:
        size = args.R
        if size < 3 or size > 100:
            print("Invalid puzzle size. Size should be between 3 and 100.")
            sys.exit(1)
        starting = generate_random_puzzle(size)
        print("Generated random puzzle:")
        printNodeText(starting)
        return starting, graphical

    else:
        print("Error: Must specify either a file [file] or the random option -R [size].")
        sys.exit(1)


def isExplored(node: Node, explored: Set):
    """
    Checks whether the given node is in the set of explored nodes.
    This function takes a node and a set of explored nodes. It returns True if
    the node is in the set, and False otherwise.
    Parameters: node (Node): The node to check.
                explored (set): The set of already explored nodes.
    Returns: bool: True if the node has been explored, False otherwise.
    """
    if node in explored:
        return True
    return False


def findGoal(goal_position):
    """
    Locates the position of the specified tile in the goal state.
    This function iterates over the tiles in the goal state to find the one
    matching the specified 'goal_position'. When it finds a match, it
    calculates and returns the (row, column) position of the tile.
    Parameters: goal_position (int): The value of the tile to locate.
    Returns: tuple: The (row, column) position of the specified tile in the goal state.
    """
    global goal
    tiles = goal.tiles
    for i in range(size * size):
        if tiles[i] == goal_position:
            return i // size, i % size


def solve(starting: Node):
    """
    Solves the puzzle using A* search algorithm.
    This function implements the A* search algorithm to find the solution
    to the puzzle. It maintains an open list (priority queue based on f-score)
    and a closed list (set of explored nodes). It starts with the initial state
    of the puzzle and explores child nodes by moving the empty tile in all
    possible directions.
    It continues this process until it finds the goal state (a solved puzzle)
    or has explored all possible states.
    Parameters: starting (Node): The initial state of the puzzle.
    Returns: Node: The final state of the puzzle if a solution is found, otherwise None.
    """
    open_list: PriorityQueue = PriorityQueue()
    open_list.put(starting)
    closed_list: Set = set()

    while len(open_list) > 0:
        best_node: Node = open_list.get()

        if best_node.is_goal():
            print("Open list length: " + str(len(open_list)))
            print("Closed list length: " + str(len(closed_list)))
            return best_node

        if isExplored(best_node, closed_list):
            continue

        children = best_node.get_children()

        for child in children:
            open_list.put(child)

        closed_list.add(best_node)

    return None


def print_solution(solution: Node, graphical: bool, time_taken: float):
    """
    Prints the solution to the puzzle.
    This function takes the final state of the puzzle and prints the solution
    to the puzzle by traversing the parent nodes from the goal state to the
    initial state.
    Parameters: solution (Node): The final state of the puzzle.
                graphical (bool): A boolean indicating if the solution should be displayed graphically.
    Returns: None
    """
    total = 0
    path = []
    while solution.parent is not None:
        path.append(solution)
        solution = solution.parent
        total += 1
    path = path[::-1]

    print("Total moves: " + str(total))
    print("Time taken: " + str(time_taken) + " seconds")

    if graphical:
        for node in path:
            printNode(node)

    else:
        with open("solution.txt", "w") as file:
            for node in path:
                printNodeText(node, file)


def main():
    starting, graphical = parseArgs()
    is_solvable = is_puzzle_solvable(np.array(starting.tiles).reshape((size, size)))
    generate_goal()
    print()
    if not is_solvable:
        print("Puzzle is not solvable.")
        sys.exit(1)
    else:
        print("Puzzle is solvable, you'll can find the steps in the solution.txt file.\n")
        print("Solving puzzle, please wait...\n")
    if graphical:
        plt.show(block=False)
    debut = time()
    solution: Node = solve(starting)
    fin = time()

    if solution is None:
        print("Puzzle is not solvable.")
        sys.exit(1)
    else:
        print_solution(solution, graphical, fin - debut)
    if graphical:
        plt.show()


if __name__ == '__main__':
    main()
