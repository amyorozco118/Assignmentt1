import math

import numpy as np
import random
import sys

boardHeight = 4
boardWidth = 4
start = "S"
slocx = 0
slocy = 0
glocx = 0
glocy = 0
goal = "G"
turned = 2
cosy = 0
didBash = False

def createBoard():
    board = []
    for i in range(boardWidth):
        row = []
        for j in range(boardHeight):
            row.append(random.randint(1, 9))
        board.append(row)

    print(board)
    slocx = random.randint(0, boardWidth - 1)
    slocy = random.randint(0, boardHeight - 1)
    glocx = random.randint(0, boardWidth - 1)
    glocy = random.randint(0, boardHeight - 1)

    print(slocx)
    print(slocy)
    print(glocx)
    print(glocy)

    # check if s and g are the same

    board[slocx][slocy] = 1
    board[glocx][glocy] = 1
    print(board)
    return board


class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


# This function return the path of the search
def return_path(current_node, maze):
    path = []
    no_rows, no_columns = np.shape(maze)
    # here we create the initialized result maze with -1 in every position
    result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node

    while current is not None:
        print("Current" , current.g)
        path.append(current.position)
        current = current.parent
    # Return reversed path as we need to show from start to end path
    path = path[::-1]

    return path

def when_bash(mapdata, x, y):
    """
    Returns the walkable 4-neighbors cells of (x,y) in the occupancy grid.
    :param mapdata [OccupancyGrid] The map information.
    :param x       [int]           The X coordinate in the grid.
    :param y       [int]           The Y coordinate in the grid.
    :return        [[(int,int)]]   A list of walkable 4-neighbors.
    """
    # Since there are only 4 points to check a "brute force" approach is sufficient
    neighbors = []

    addy = [2, -2, 0, 0]
    addx = [0, 0, 2, -2]
    for i in range(len(addx)):
        cols = 3
        rows = 3

        if cols - 1 > y + addy[i] and \
                rows - 1 > x + addx[i] and \
                0 <= x + addx[i] and \
                0 <= y + addy[i]:
            neighbors.append((x + addx[i], y + addy[i]))

    return neighbors

def search(maze, cost, start, end, heur):
    """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param cost
        :param start:
        :param end:
        :return:
    """

    cc = 0
    nodes = 0
    h = heur
    # Create start and end node with initized values for g, h and f
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both yet_to_visit and visited list
    # in this list we will put all node that are yet_to_visit for exploration.
    # From here we will find the lowest cost node to expand next
    yet_to_visit_list = []
    # in this list we will put all node those already explored so that we don't explore it again
    visited_list = []

    # Add the start node
    yet_to_visit_list.append(start_node)

    # Adding a stop condition. This is to avoid any infinite loop and stop
    # execution after some reasonable number of steps
    # outer_iterations = 0
    # max_iterations = (len(maze) // 2) ** 10

    # what squares do we search . serarch movement is left-right-top-bottom
    # (4 movements) from every positon
    turned = 2
    # move = []
    # isBash = False
    # if isBash == False:
    #     move = [[-1, 0],  # go up
    #             [0, -1],  # go left
    #             [1, 0],  # go down
    #             [0, 1]]  # go right
    # else:
    #     move[-1, 0]






    """
        1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
        2) Check max iteration reached or not . Set a message and stop execution
        3) Remove the selected node from yet_to_visit list and add this node to visited list
        4) Perofmr Goal test and return the path else perform below steps
        5) For selected node find out all children (use move to find children)
            a) get the current postion for the selected node (this becomes parent node for the children)
            b) check if a valid position exist (boundary will make few nodes invalid)
            c) if any node is a wall then ignore that
            d) add to valid children node list for the selected parent

            For all the children node
                a) if child in visited list then ignore it and try next node
                b) calculate child node g, h and f values
                c) if child in yet_to_visit list then ignore it
                d) else move the child to yet_to_visit list
    """
    # find maze has got how many rows and columns
    no_rows, no_columns = np.shape(maze)
    print(no_rows)

    # Loop until you find the end
    x = 0

    while len(yet_to_visit_list) > 0:
        move = []
        # isBash = False
        # if isBash == False:
        #     move = [[-1, 0],  # go up
        #             [0, -1],  # go left
        #             [1, 0],  # go down
        #             [0, 1]]  # go right
        # else:
        #     if(turned == 2):
        #         move[-1, 0]
        #     elif turned == -2:
        #         move[1,0]
        #     elif turned == -1:
        #         move[0,-1]
        #     elif turned == 1:
        #         move[0,1]
        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        # outer_iterations += 1

        move = [[-1, 0],  # go up
                            [0, -1],  # go left
                            [1, 0],  # go down
                            [0, 1]]  # go right
        # Get the current node
        current_node = yet_to_visit_list[0]


        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # if we hit this point return the path such as it may be no solution or
        # computation cost is too high
        # if outer_iterations > max_iterations:
        #     print("giving up on pathfinding too many iterations")
        #     return return_path(current_node, maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path

        if len(visited_list) > 1:
            x = x+1
            direction = getDirection(current_node, turned)
        else:
            direction = 2
        turned = direction
        print("Direction",turned)
        if current_node == end_node:
            print("Nodes", nodes)
            return return_path(current_node, maze)

        # Generate children from all adjacent squares
        bash = []

        moveBash = when_bash(maze, current_node.position[0], current_node.position[1])
        for new_position in moveBash:

            # Get node position
            node_position = (new_position[0],new_position[1])

            # Make sure within range (check if within maze boundary)
            if (node_position[0] > (no_rows - 1) or
                    node_position[0] < 0 or
                    node_position[1] > (no_columns - 1) or
                    node_position[1] < 0):
                continue

            # Make sure walkable terrain
            # if maze[node_position[0]][node_position[1]] != 0:
            #     continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            bash.append(new_node)

            # Generate children from all adjacent squares
        children = []

        for new_position in move:

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range (check if within maze boundary)
                if (node_position[0] > (no_rows - 1) or
                        node_position[0] < 0 or
                        node_position[1] > (no_columns - 1) or
                        node_position[1] < 0):
                    continue

                # Make sure walkable terrain
                # if maze[node_position[0]][node_position[1]] != 0:
                #     continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)
        # Loop through children

        for child in children:

            isBash = False
            # Child is on the visited list (search entire visited list)
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue


            bashVal = bash_total(child, current_node, maze, direction)
            extraVal = extraValue(child, current_node, maze, direction)
            if (extraVal > bashVal):
                child.g = current_node.g + bashVal
                didBash = True
                print("Cost of g: ", child.g)
            else:
                child.g = current_node.g + extraVal
                print("Cost of g: ", child.g)

            ## Heuristic costs calculated here, this is using eucledian distance
            child.h = getHeur(heur, current_node, end_node, child)
            print(child.h)

            child.f = child.g + child.h
            nodes = nodes + 1
            print("Cost",child.f)
            # Child is already in the yet_to_visit list and g cost is already lower
            if len([i for i in yet_to_visit_list if child == i and child.f > i.f]) > 0:

                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_list.append(child)
            print(child.position)

        # for child in bash:
        #     isBash = True
        #     if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
        #         continue
        #     child.g = 3 + maze[child.position[0]][child.position[1]]  # deal with bash by getting next move
        #     child.h = 0
        #     child.f = child.g + child.h
        #
        #     nodes = nodes + 1
        #     if len([i for i in yet_to_visit_list if child == i and child.f > i.f]) > 0:
        #
        #         continue
        #
        #     # Add the child to the yet_to_visit list
        #     yet_to_visit_list.append(child)
        #     print(child.position)

def bash_total(child, end_node, map, dire):
    childx = child.position[0]
    childy = child.position[1]

    print("End Node position: " + str(end_node.position[0]) + ", " + str(end_node.position[1]))
    print("Child Node position: " + str(childx) + ", " + str(childy))


    totalCost = 0
    turned = dire
    # one turn
    # Moving to the right
    if childy == end_node.position[1] + 1:
        if turned == 1:
            totalCost = 3 + map[childx][childy - 1]
            return math.ceil(totalCost)
        if turned == 2:
            totalCost = .5 * map[childx][childy] + 3 + map[childx][childy - 1]
            turned = 1
            return math.ceil(totalCost)
        if turned == -1:
            totalCost = .5 * map[childx][childy] + .5 * map[childx][childy] + 3 + map[childx][childy - 1]
            turned = 1
            return math.ceil(totalCost)
        if turned == -2:
            totalCost = .5 * map[childx][childy] + 3 + map[childx][childy - 1]
            turned = 1
            return math.ceil(totalCost)
    # Moving to the left
    if childy == end_node.position[1] - 1:
        if turned == 1:
            totalCost = .5 * map[childx][childy] + .5 * map[childx][childy] + 3 + map[childx][childy + 1]
            turned = -1
            return math.ceil(totalCost)
        if turned == 2:
            totalCost = .5 * map[childx][childy] + 3 + map[childx][childy + 1]
            turned = -1
            return math.ceil(totalCost)
        if turned == -1:
            totalCost =  3 + map[childx][childy + 1]
            turned = -1
            return math.ceil(totalCost)
        if turned == -2:
            totalCost = .5 * map[childx][childy] + 3 + map[childx][childy + 1]
            turned = -1
            return math.ceil(totalCost)
    # Going Forward
    if childx == end_node.position[0] - 1:
        if turned == 1:
            totalCost = .5 * map[childx][childy] + 3 + map[childx + 1][childy]
            turned = 2
            return math.ceil(totalCost)
        if turned == 2:
            totalCost = 3 + map[childx + 1][childy]
            turned = 2
            return math.ceil(totalCost)
        if turned == -1:
            totalCost = .5 * map[childx][childy] + 3 + map[childx + 1][childy]
            turned = 2
            return math.ceil(totalCost)
        if turned == -2:
            totalCost = .5 * map[childx][childy] + .5 * map[childx][childy] + 3 + map[childx + 1][childy]
            turned = 2
            return math.ceil(totalCost)
    # Going Backwards
    if childx == end_node.position[0] + 1:
        if turned == 1:
            totalCost = .5 * map[childx][childy] + 3 + map[childx - 1][childy]
            turned = -2
            return math.ceil(totalCost)
        if turned == 2:
            totalCost = .5 * map[childx][childy] + .5 * map[childx][childy] + 3 + map[childx - 1][childy]
            turned = -2
            return math.ceil(totalCost)
        if turned == -1:
            totalCost = .5 * map[childx][childy] + 3 + map[childx - 1][childy]
            turned = -2
            return math.ceil(totalCost)
        if turned == -2:
            totalCost = 3 + map[childx - 1][childy]
            turned = -2
            return math.ceil(totalCost)
    else:
        return 0



def getDirection(child, turned):
    childx = child.position[0]
    childy = child.position[1]

    end_node = child.parent
    if childy == end_node.position[1] + 1:
        if turned == 1:
            return turned
        if turned == 2:
            turned = 1
            return turned
        if turned == -1:
            turned = 1
            return turned
        if turned == -2:
            turned = 1
            return turned
        # Moving to the left
    if childy == end_node.position[1] - 1:
        if turned == 1:
            turned = -1
            return turned
        if turned == 2:
            turned = -1
            return turned
        if turned == -1:
            turned = -1
            return turned
        if turned == -2:
            turned = -1
            return turned
        # Going Forward
    if childx - 1 == end_node.position[0]:
        if turned == 1:
            turned = 2
            return turned
        if turned == 2:
            turned = 2
            return turned
        if turned == -1:
            turned = 2
            return turned
        if turned == -2:
            turned = 2
            return turned
        # Going Backwards
    if childx == end_node.position[0] + 1:
        if turned == 1:
            turned = -2
            return turned
        if turned == 2:
            turned = -2
            return turned
        if turned == -1:
            turned = -2
            return turned
        if turned == -2:
            turned = -2
            return turned
    else:
        return turned


def extraValue(child, end_node, map, dire):
    childx = child.position[0]
    childy = child.position[1]

    totalCost = 0
    turned = dire
    # one turn
    # Moving to the right
    if childy == end_node.position[1] + 1:
        if turned == 1:
            totalCost = map[end_node.position[0]][end_node.position[1]]
            return math.ceil(totalCost)
        if turned == 2:
            totalCost = .5 * map[childx][childy] + map[end_node.position[0]][end_node.position[1]]
            turned = 1
            return math.ceil(totalCost)
        if turned == -1:
            totalCost = .5 * map[childx][childy] + .5 * map[childx][childy] + map[end_node.position[0]][
                end_node.position[1]]
            turned = 1
            return math.ceil(totalCost)
        if turned == -2:
            totalCost = .5 * map[childx][childy] + map[end_node.position[0]][end_node.position[1]]
            turned = 1
            return math.ceil(totalCost)
    # Moving to the left
    if childy == end_node.position[1] - 1:
        if turned == 1:
            totalCost = .5 * map[childx][childy] + .5 * map[childx][childy] + map[end_node.position[0]][
                end_node.position[1]]
            turned = -1
            return math.ceil(totalCost)
        if turned == 2:
            totalCost = .5 * map[childx][childy] + map[end_node.position[0]][end_node.position[1]]
            turned = -1
            return math.ceil(totalCost)
        if turned == -1:
            totalCost =  map[end_node.position[0]][
                end_node.position[1]]
            turned = -1
            return math.ceil(totalCost)
        if turned == -2:
            totalCost = .5 * map[childx][childy] + map[end_node.position[0]][end_node.position[1]]
            turned = -1
            return math.ceil(totalCost)
    # Going Forward
    if childx == end_node.position[0] - 1:
        if turned == 1:
            totalCost = .5 * map[childx][childy] + map[end_node.position[0]][end_node.position[1]]
            turned = 2
            return math.ceil(totalCost)
        if turned == 2:
            totalCost = map[childx][childy]
            turned = 2
            return math.ceil(totalCost)
        if turned == -1:
            totalCost = .5 * map[childx][childy] + map[end_node.position[0]][end_node.position[1]]
            turned = 2
            return math.ceil(totalCost)
        if turned == -2:
            totalCost = .5 * map[childx][childy] + .5 * map[childx][childy] + map[end_node.position[0]][
                end_node.position[1]]
            turned = 2
            return math.ceil(totalCost)
    # Going Backwards
    if childx == end_node.position[0] + 1:
        if turned == 1:
            totalCost = .5 * map[childx][childy] + map[end_node.position[0]][end_node.position[1]]
            turned = -2
            return math.ceil(totalCost)
        if turned == 2:
            totalCost = .5 * map[childx][childy] + .5 * map[childx][childy] + map[end_node.position[0]][
                end_node.position[1]]
            turned = -2
            return math.ceil(totalCost)
        if turned == -1:
            totalCost = .5 * map[childx][childy] + map[end_node.position[0]][end_node.position[1]]
            turned = -2
            return math.ceil(totalCost)
        if turned == -2:
            totalCost = map[end_node.position[0]][end_node.position[1]]
            turned = -2
            return math.ceil(totalCost)
    else:
        return 0

def parseImportedBoard(filename):
    # Tab delimited file input to board
    maze = []
    row = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            lineArray = []
            split = line.strip().split("\t")
            col = 0
            for number in split:
                if (number == 'G'):
                    goal = [row, col]
                    lineArray.append(0)
                elif (number == 'S'):
                    print(str(row) +"," + str(col))
                    start = [row, col]
                    lineArray.append(0)
                else:
                    lineArray.append(int(number))
                col = col + 1
            maze.append(lineArray)
            row = row + 1
    return maze, start, goal
def when_bash(mapdata, x, y):
    """
    Returns the walkable 4-neighbors cells of (x,y) in the occupancy grid.
    :param mapdata [OccupancyGrid] The map information.
    :param x       [int]           The X coordinate in the grid.
    :param y       [int]           The Y coordinate in the grid.
    :return        [[(int,int)]]   A list of walkable 4-neighbors.
    """
    # Since there are only 4 points to check a "brute force" approach is sufficient
    neighbors = []

    addy = [2, -2, 0, 0]
    addx = [0, 0, 2, -2]
    for i in range(len(addx)):
        cols = 3
        rows = 3

        if cols - 1 > y + addy[i] and \
                rows - 1 > x + addx[i] and \
                0 <= x + addx[i] and \
                0 <= y + addy[i]:
            neighbors.append((x + addx[i], y + addy[i]))

    return neighbors


def parseImportedBoard(filename):
    # Tab delimited file input to board
    maze = []
    row = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            lineArray = []
            split = line.strip().split("\t")
            col = 0
            for number in split:
                if (number == 'G'):
                    goal = [row, col]
                    lineArray.append(0)
                elif (number == 'S'):
                    print(str(row) + "," + str(col))
                    start = [row, col]
                    lineArray.append(0)
                else:
                    lineArray.append(int(number))
                col = col + 1
            maze.append(lineArray)
            row = row + 1
    return maze, start, goal

def getHeur(heur, current, goal, next):
        vertDis = abs(goal.position[1] - next.position[1])
        horDist = abs(goal.position[0] - next.position[0])

        cost_so_far = 0
        if heur == 1:
            heurCost = current.h
            return heurCost
        if heur == 2:
            heurCost = current.h + min(vertDis, horDist)
            return heurCost
        if heur == 3:
            heurCost = current.h + max(vertDis, horDist)
            return heurCost
        if heur == 4:
            heurCost = current.h + vertDis + horDist
            return heurCost

if __name__ == '__main__':
    #heristicToUse = sys.argv[1]
    # if (len(sys.argv) > 1):
    #     maze, start, end = parseImportedBoard(sys.argv[2])  # returns board from imported file
    # else:
        #maze = createBoard()
    maze = [[4, 1, 4, 6], [2, 9, 9, 6], [1, 1, 1, 3]]
    for row in maze:
        print(row)
    start = (2, 2)  # starting position
    end = (0, 1)  # ending position

    cost = 1  # cost per movement

    path = search(maze, cost, start, end, 1)
    print(path)