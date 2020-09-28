from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from csv import reader
from math import sqrt
import random
import pygame
import random
import sys
import math



# region SearchAlgorithms
class Node:
    id = None  # Unique value for each node.
    up = None  # Represents value of neighbors (up, down, left, right).
    down = None
    left = None
    right = None
    previousNode = None  # Represents value of neighbors.
    def __init__(self, value):
        self.value = value


class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    strMaze = []
    visited = []
    start_node=None
    allMazeNodes=[]
    stack=[]
    bool=0
    def __init__(self, mazeStr):
        ''' mazeStr contains the full board
         The board is read row wise,
        the nodes are numbered 0-based starting
        the leftmost node'''
        row = mazeStr.split(' ')
        for j in row:
            rowelements = j.split(',')
            self.strMaze.append(rowelements)
##################################################Split maze String into 2d nodes
        for i in range(len(self.strMaze)):
            oneRow=[]
            for j in range(len(self.strMaze[i])):
                node = Node(self.strMaze[i][j])
                node.id=[i,j]
                if i-1>=0 :
                    node.up=[i-1,j]
                if i + 1 < len(self.strMaze):
                    node.down=[i+1,j]
                if j-1>=0:
                    node.left=[i,j-1]
                if j+1<len(self.strMaze[i]):
                    node.right=[i,j+1]
                oneRow.append(node)
            self.allMazeNodes.append(oneRow)
        ''' for i in self.allMazeNodes:
            self.notVisited.append(i)'''
###########################Search for the start node
        for i in range(len(self.allMazeNodes)):
            for j in range(len(self.allMazeNodes[i])):
                if self.allMazeNodes[i][j].value=='S':
                    self.start_node=self.allMazeNodes[i][j]
        ##self.fullPath.append(self.start_node)
    def Backtrack(self,myStartNode):
        while myStartNode is not None:
            self.path.append(myStartNode.id[0] * len(self.allMazeNodes[0]) + myStartNode.id[1])##Convert 2d indx to 1d Index
            myStartNode = myStartNode.previousNode
        self.path.reverse()
        return myStartNode
    def dfsHelper(self):
        self.stack.append(self.start_node)
        self.visited.append(self.start_node)
        while len(self.stack)>0:
            myStartNode =self.stack[0]
            self.fullPath.append(myStartNode.id[0] * len(self.allMazeNodes[0]) +myStartNode.id[1])
            self.stack.remove(self.stack[0])
            if myStartNode.value=="E":
                myStartNode =self.Backtrack(myStartNode)
                break
            left = myStartNode.left
            down = myStartNode.down
            right = myStartNode.right
            up = myStartNode.up
            ##Up move
            if myStartNode.up != None and self.allMazeNodes[up[0]][up[1]] not in self.visited and self.allMazeNodes[up[0]][up[1]].value != '#':
                self.stack.insert(0,self.allMazeNodes[up[0]][up[1]])
                self.visited.append(self.allMazeNodes[up[0]][up[1]])
                upNode=self.allMazeNodes[up[0]][up[1]]
                upNode.previousNode=myStartNode
                myStartNode.up=upNode.id
            ##Left Move
            if myStartNode.left != None and self.allMazeNodes[left[0]][left[1]] not in self.visited and self.allMazeNodes[left[0]][left[1]].value != '#':
                self.stack.insert(0, self.allMazeNodes[left[0]][left[1]])
                self.visited.append(self.allMazeNodes[left[0]][left[1]])
                leftNode = self.allMazeNodes[left[0]][left[1]]
                leftNode.previousNode = myStartNode
                myStartNode.left = leftNode.id
            ##Right Move
            if myStartNode.right != None and self.allMazeNodes[right[0]][right[1]] not in self.visited and self.allMazeNodes[right[0]][right[1]].value != '#':
                self.stack.insert(0, self.allMazeNodes[right[0]][right[1]])
                self.visited.append(self.allMazeNodes[right[0]][right[1]])
                rightNode = self.allMazeNodes[right[0]][right[1]]
                rightNode.previousNode = myStartNode
                myStartNode.right = rightNode.id
            ##Down Move
            if myStartNode.down != None and self.allMazeNodes[down[0]][down[1]] not in self.visited and self.allMazeNodes[down[0]][down[1]].value != '#':
                self.stack.insert(0, self.allMazeNodes[down[0]][down[1]])
                self.visited.append(self.allMazeNodes[down[0]][down[1]])
                downNode = self.allMazeNodes[down[0]][down[1]]
                downNode.previousNode = myStartNode
                myStartNode.down = downNode.id


        return self.fullPath, self.path
    def DFS(self):
        # Fill the correct path in self.path
        # self.fullPath should contain the order of visited nodes
        # self.path should contain the direct path from start node to goal node
        ##return self.fullPath, dfsHelper(self.fullPath[0])
        return self.dfsHelper()




# endregion

#region Gaming
class Gaming:
    def __init__(self):
        self.COLOR_BLUE = (0, 0, 240)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_RED = (255, 0, 0)
        self.COLOR_YELLOW = (255, 255, 0)

        self.Y_COUNT = int(5) ## Num of Column
        self.X_COUNT = int(8)

        self.PLAYER = 0 ## wich player will play
        self.AI = 1

        self.PLAYER_PIECE = 1 ##id pics
        self.AI_PIECE = 2

        self.WINNING_WINDOW_LENGTH = 3   ## connect three to win
        self.EMPTY = 0
        self.WINNING_POSITION = []
        self.SQUARESIZE = 80

        self.width = self.X_COUNT * self.SQUARESIZE
        self.height = (self.Y_COUNT + 1) * self.SQUARESIZE

        self.size = (self.width, self.height)

        self.RADIUS = int(self.SQUARESIZE / 2 - 5)

        self.screen = pygame.display.set_mode(self.size)
    def create_board(self):
        board = np.zeros((self.Y_COUNT, self.X_COUNT))
        return board


    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece


    def is_valid_location(self, board, col):  ## for not making rePlaying for col
        return board[self.Y_COUNT - 1][col] == 0


    def get_next_open_row(self, board, col):  ## for not making rePlaying for row
        for r in range(self.Y_COUNT):
            if board[r][col] == 0:
                return r


    def print_board(self, board):
        print(np.flip(board, 0))


    def winning_move(self, board, piece):    ##select the format of winnig player
        self.WINNING_POSITION.clear()
        for c in range(self.X_COUNT - 2):
            for r in range(self.Y_COUNT):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r, c + 1])
                    self.WINNING_POSITION.append([r, c + 2])
                    return True

        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT - 2):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r + 1, c])
                    self.WINNING_POSITION.append([r + 2, c])
                    return True

        for c in range(self.X_COUNT - 2):
            for r in range(self.Y_COUNT - 2):
                if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r + 1, c + 1])
                    self.WINNING_POSITION.append([r + 2, c + 2])
                    return True

        for c in range(self.X_COUNT - 2):
            for r in range(2, self.Y_COUNT):
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r - 1, c + 1])
                    self.WINNING_POSITION.append([r - 2, c + 2])
                    return True


    def evaluate_window(self, window, piece):  ## maark window with points
        score = 0
        opp_piece = self.PLAYER_PIECE
        if piece == self.PLAYER_PIECE:
            opp_piece = self.AI_PIECE

        if window.count(piece) == 3:
            score += 100
        elif window.count(piece) == 2 and window.count(self.EMPTY) == 1:
            score += 5

        if window.count(opp_piece) == 3 and window.count(self.EMPTY) == 1:
            score -= 4

        return score

    def draw_board(self, board):
        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT):
                pygame.draw.rect(self.screen, self.COLOR_BLUE,
                                 (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE,
                                  self.SQUARESIZE))
                pygame.draw.circle(self.screen, self.COLOR_BLACK, (
                        int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                        int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)),
                                   self.RADIUS)

        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT):
                if board[r][c] == self.PLAYER_PIECE:
                    pygame.draw.circle(self.screen, self.COLOR_RED, (
                            int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                            self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)),
                                       self.RADIUS)
                elif board[r][c] == self.AI_PIECE:
                    pygame.draw.circle(self.screen, self.COLOR_YELLOW, (
                            int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                            self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)),
                                       self.RADIUS)
        pygame.display.update()

    def score_position(self, board, piece):
        score = 0

        center_array = [int(i) for i in list(board[:, self.X_COUNT // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        for r in range(self.Y_COUNT):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(self.X_COUNT - 3):
                window = row_array[c: c + self.WINNING_WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for c in range(self.X_COUNT):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(self.Y_COUNT - 3):
                window = col_array[r: r + self.WINNING_WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for r in range(self.Y_COUNT - 3):
            for c in range(self.X_COUNT - 3):
                window = [board[r + i][c + i] for i in range(self.WINNING_WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(self.Y_COUNT - 3):
            for c in range(self.X_COUNT - 3):
                window = [board[r + 3 - i][c + i] for i in range(self.WINNING_WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score


    def is_terminal_node(self, board):  ##check if Ai is Win or human or the game is game over
        return self.winning_move(board, self.PLAYER_PIECE) or self.winning_move(board, self.AI_PIECE) or len(self.get_valid_locations(board)) == 0

    def terminal(self,is_terminal,board):  ##terminal game and return the result for winning player
        if is_terminal:
            if self.winning_move(board, self.PLAYER_PIECE):
                return (None, -99999999)          ##Human is win
            elif self.winning_move(board, self.AI_PIECE):
                return (None, 99999999)             ##Ai is Win
            else:
                return (None, -1)  # here game is over
        else:
            return (None, self.score_position(board, self.AI_PIECE))  # last element depth

    staticNumOFColumn=1
    def AlphaBeta(self, board, depth, alpha, beta, currentPlayer):
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)
        value = -math.inf ## negative to minimize
        column = random.choice(valid_locations)
        '''Implement here'''
        if depth == -1 or is_terminal:
            return self.terminal( is_terminal, board)
        if currentPlayer:  ##Ai will work
            for col in valid_locations:
                b_copy = board.copy()
                row = self.get_next_open_row(board, col)
                self.drop_piece(b_copy, row, col, self.AI_PIECE)
                new_score = self.AlphaBeta(b_copy, depth - 1, alpha, beta, False)[self.staticNumOFColumn] ##call to minimize human play
                if new_score > value:
                    column = col
                    value = new_score
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column,value
        else:
            value *= -1                                                           #here you need to minimize the player 2 role
            for col in valid_locations:
                b_copy = board.copy()
                row = self.get_next_open_row(board, col)
                self.drop_piece(b_copy, row, col, self.PLAYER_PIECE)
                new_score = self.AlphaBeta(b_copy, depth - 1, alpha, beta, True)[self.staticNumOFColumn] ##call to maximze Ai play
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_valid_locations(self, board): ##return all valid locations in List
        valid_locations = []
        for col in range(self.X_COUNT):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations
    def pick_best_move(self, board, piece):  ##like mini max
        best_score = -10000
        valid_locations = self.get_valid_locations(board)
        best_col = random.choice(valid_locations)

        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = board.copy()
            self.drop_piece(temp_board, row, col, piece)
            score = self.score_position(temp_board, piece)

            if score > best_score:
                best_score = score
                best_col = col

        return best_col



#endregion

# region KMEANS
class DataItem:
    def __init__(self, item):
        self.features = item ## X , Y
        self.clusterId = -1

    def getDataset(self):     ##Fill Dataset
        data = []

        return data.append(DataItem([0, 0, 0, 0]))
        data.append(DataItem([0, 0, 0, 1]))
        data.append(DataItem([0, 0, 1, 0]))
        data.append(DataItem([0, 0, 1, 1]))
        data.append(DataItem([0, 1, 0, 0]))
        data.append(DataItem([0, 1, 0, 1]))
        data.append(DataItem([0, 1, 1, 0]))
        data.append(DataItem([0, 1, 1, 1]))
        data.append(DataItem([1, 0, 0, 0]))
        data.append(DataItem([1, 0, 0, 1]))
        data.append(DataItem([1, 0, 1, 0]))
        data.append(DataItem([1, 0, 1, 1]))
        data.append(DataItem([1, 1, 0, 0]))
        data.append(DataItem([1, 1, 0, 1]))
        data.append(DataItem([1, 1, 1, 0]))
        data.append(DataItem([1, 1, 1, 1]))

class Cluster:
        def __init__(self, id, centroid): ## first centroid will be data point
            self.centroid = centroid
            self.data = []
            self.id = id

        def update(self, clusterData):  ## Update center to get real centroid
            self.data = []
            for item in clusterData:
                self.data.append(item.features)
            tmpC = np.average(self.data, axis=0)  ## To take average for each column
            tmpL = []
            for i in tmpC:
                tmpL.append(i)
            self.centroid = tmpL

class SimilarityDistance:
        def euclidean_distance(self, p1, p2):
            sum = 0
            for i in range(len(p1)):
                sum += (p1[i] - p2[i]) ** 2     ##euclidean_distance
            return sqrt(sum)

        def Manhattan_distance(self, p1, p2):
            sum = 0
            for i in range(len(p1)):
                sum += abs(p1[i] - p2[i])           ##Manhattan_distance
            return sum

class Clustering_kmeans:
        def __init__(self, data, k, noOfIterations, isEuclidean):
            self.noOfIterations = noOfIterations
            self.k = k     ## number of cluster that you need to bee output
            self.data = data
            self.distance = SimilarityDistance()
            self.isEuclidean = isEuclidean

        def initClusters(self):
            self.clusters = []
            for i in range(self.k):
                self.clusters.append(Cluster(i, self.data[i * 10].features))

        def getClusters(self):
            self.initClusters()
            '''Implement Here'''
            for iteration in range(self.noOfIterations):
                for item in self.data:
                    minDistance = 999999
                    for cluster in  range(self.k):
                        if self.isEuclidean==1:     ##euclidean_distance
                            finalResalt = self.distance.euclidean_distance(self.clusters[cluster].centroid, item.features)
                        else:                       ##Manhattan_distance
                            finalResalt = self.distance.Manhattan_distance(self.clusters[cluster].centroid, item.features)
                        clusterDistance=finalResalt
                        if (clusterDistance < minDistance):
                            item.clusterId = self.clusters[cluster].id
                            minDistance = clusterDistance
                    clusterData = [x for x in self.data if x.clusterId == item.clusterId]
                    self.clusters[item.clusterId].update(clusterData)  ##get the real centroid
            return self.clusters
# endregion

#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn
def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    fullPath, path = searchAlgo.DFS()
    print('**DFS**\n Full Path is: ' + str(fullPath) +'\n Path is: ' + str(path))

# endregion

#region Gaming_Main_fn
def Gaming_Main():
    game = Gaming()
    board = game.create_board()
    game.print_board(board)
    game_over = False

    pygame.init()

    game.draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 50)

    turn = 1

    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(game.screen, game.COLOR_BLACK, (0, 0, game.width, game.SQUARESIZE))
                posx = event.pos[0]
                if turn == game.PLAYER:
                    pygame.draw.circle(game.screen, game.COLOR_RED, (posx, int(game.SQUARESIZE / 2)), game.RADIUS)

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(game.screen, game.COLOR_BLACK, (0, 0, game.width, game.SQUARESIZE))

                if turn == game.PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / game.SQUARESIZE))

                    if game.is_valid_location(board, col):
                        row = game.get_next_open_row(board, col)
                        game.drop_piece(board, row, col, game.PLAYER_PIECE)

                        if game.winning_move(board, game.PLAYER_PIECE):
                            label = myfont.render("Player Human wins!", 1, game.COLOR_RED)
                            print(game.WINNING_POSITION)
                            game.screen.blit(label, (40, 10))
                            game_over = True

                        turn += 1
                        turn = turn % 2

                        # game.print_board(board)
                        game.draw_board(board)

        if turn == game.AI and not game_over:

            col, minimax_score = game.AlphaBeta(board, 5, -math.inf, math.inf, True)

            if game.is_valid_location(board, col):
                row = game.get_next_open_row(board, col)
                game.drop_piece(board, row, col, game.AI_PIECE)

                if game.winning_move(board, game.AI_PIECE):
                    label = myfont.render("Player AI wins!", 1, game.COLOR_YELLOW)
                    print(game.WINNING_POSITION)
                    game.screen.blit(label, (40, 10))
                    game_over = True

                # game.print_board(board)
                game.draw_board(board)

                turn += 1
                turn = turn % 2

        if game_over:
            pygame.time.wait(3000)
            return game.WINNING_POSITION
#endregion


# region KMeans_Main_Fn
def Kmeans_Main():
    dataset = DataItem.getDataset(None)
    # 1 for Euclidean and 0 for Manhattan
    clustering = Clustering_kmeans(dataset, 2, len(dataset),1)
    clusters = clustering.getClusters()
    for cluster in clusters:
        for i in range(4):
            cluster.centroid[i] = round(cluster.centroid[i], 2)
        print(cluster.centroid[:4])
    return clusters

# endregion


######################## MAIN ###########################33
if __name__ == '__main__':

    SearchAlgorithm_Main()
    Gaming_Main()
    Kmeans_Main()
