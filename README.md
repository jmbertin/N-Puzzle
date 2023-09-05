# N-Puzzle Solver
N-Puzzle Solver is a project undertaken as part of the 42 school curriculum. This is a Python script for solving the N-Puzzle problem using the A* search or greedy algorithms. The puzzle size is configurable and you can use different heuristic functions (Manhattan, Euclidean, Hamming, and Linear conflict / Manhattan) to calculate the distance to the goal state.

**Developed and tested on a Linux Ubuntu 23.04.**

----

## Features
- **Configurable Puzzle Size**: The solver is designed to handle puzzles of varying sizes, from the classic 3x3 grid to larger challenges, up to 100x100.
- **Multiple Heuristic Functions**: Choose from a range of heuristic functions to guide the solver:
    - Manhattan Distance: Calculates the sum of the absolute values of horizontal and vertical distances between the current state and the goal.
    - Euclidean Distance: Measures the straight-line distance between the current state and the goal.
    - Hamming Distance: Counts the number of misplaced tiles.
    - Linear Conflict + Manhattan: A combination of Manhattan distance and additional penalties for tiles in linear conflict.
- **Algorithm Choice**: Decide between the A* search algorithm, known for its efficiency and accuracy, or the Greedy algorithm, which can be faster but might not always find the shortest path.
- **Graphical Display**: For puzzles up to 25x25 in size, view the solution graphically, watching each step as the puzzle pieces move into place.
- **Random Puzzle Generation**: Don't have a puzzle on hand? Generate a random one of your desired size. Ensure it's solvable or unsolvable based on your preference.
- **Integrated Puzzle Generator**: Apart from the main solver, an integrated puzzle generator (generator.py) is provided to create custom puzzles. This allows for more control over the puzzle's difficulty and solvability.
- **Solution Output**: After solving, view the solution steps in a dedicated solution.txt file or watch them play out graphically.
- **User-Friendly Command-Line Interface**: The solver offers a range of command-line options for customization, ensuring you have the tools you need to tackle any puzzle.

----

## Installation
```
git clone https://github.com/Apyre83/n-puzzle
cd n-puzzle
pip install -r requirements.txt
```

----

## Usage

``npuzzle.py [-h] [-G] [-H {manhattan,lconflict,euclidean,hamming}] [-R [3-100]] [-A {astar,greedy}] [file]``

**positional arguments:**
**[file]** - The puzzle file to solve, mandatory without -R option.

**options:**
**-h, --help** - Show help message and exit
**-G** - Display solution graphically (size 25 max).
**-H {manhattan,lconflict,euclidean,hamming}** - Choose the heuristic function (default: manhattan)
**-R [3-100]** - Generate a random puzzle of the specified size (3-100).
**-A {astar,greedy}** - Choose the search algorithm (default: astar)

----

## Using generator to create a new n-puzzle

First, if you choose not to use **-R option** of the npuzzle.py, you'll have to generate a puzzle by using the generator.py script :

```generator.py [size] [-i [iteration]] [-s] or [-u] > new_puzzle```

**positional arguments:**
**[size]** - The size of the puzzle to generate [3-x]. Minimum size is 3, no maximum. Exemple size 5 means a n-puzzle 5x5.

**options:**
**-s** - (Optional Flag) If this option is used, the script will generate a puzzle that is guaranteed to be solvable
**-u** - (Optional Flag) If this option is used, the script will generate a puzzle that is guaranteed to be unsolvable
**-i [iteration]** - (Optional Argument) The number of moves to make when shuffling the puzzle. The higher the number of iterations, the more shuffled the generated puzzle will be. By default, this number is set to 1000.

----

## Solving a puzzle

Once you have a puzzle in a file, you are ready to solve it using the generator.py script :

``python npuzzle.py new_puzzle``

If you want to generate a new n-puzzle by the integrated generator and solve it directly :
``python npuzzle.py -R [size]``

Different options are available, find them in the **usage** section of this document or by using the -h option :
``python n-puzzle.py -h``

Once the puzzle is solved, if you've specified the -G option, the solution will be displayed in a new window (10 steps / second), else, the steps of the resolution will be available in the solution.txt file.

This project was made with [Python 3.11.2](https://www.python.org/downloads/release/python-3112/)

----

**Authors are:**
- [Apyre / Leo Fresnay](https://github.com/Apyre83)
- [Jean-michel Bertin](https://github.com/jmbertin)
