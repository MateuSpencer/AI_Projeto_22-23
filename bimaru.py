# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 100032 Mateus Spencer
# 95832 Miguel Cunha

import sys
import numpy as np
import copy
from typing import Tuple
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class BimaruState:
    state_id = 0
    
    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id


class Board:
    """Representação interna de um tabuleiro de Bimaru."""
    
    def __init__(self, board, row_pieces_placed, col_pieces_placed, remaining_pieces, bimaru):
        self.board = board
        self.bimaru = bimaru # the bimaru problem object, to access the rows & columns hints
        self.remaining_pieces = remaining_pieces # total number of pieces to be placed
        self.row_pieces_placed = row_pieces_placed # vector to store how many pieces have been placed in each row
        self.col_pieces_placed = col_pieces_placed # vector to store how many pieces have been placed in each column
        self.fill_completed_row_col()

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row][col]

    def set_value(self, row: int, col: int, value: str):
        """Define o valor na respetiva posição do tabuleiro."""
        self.board[row][col] = value

    def adjacent_vertical_values(self, row: int, col: int) -> Tuple[str, str]:
        """Devolve os valores imediatamente acima e abaixo, respectivamente."""
        above = self.get_value(row-1, col) if row > 0 else ""
        below = self.get_value(row+1, col) if row < 9 else ""
        return above, below

    def adjacent_horizontal_values(self, row: int, col: int) -> Tuple[str, str]:
        """Devolve os valores imediatamente à esquerda e à direita, respectivamente."""
        left = self.get_value(row, col-1) if col > 0 else ""
        right = self.get_value(row, col+1) if col < 9 else ""
        return left, right


    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        row_hints = []
        col_hints = []
        remaining_pieces = {"C": 4, "M": 4, "TBRL": 12} # initial number of pieces to be placed on the board
        board = np.zeros((10, 10), dtype=str)
        row_pieces_placed = np.zeros(10, dtype=int)
        col_pieces_placed = np.zeros(10, dtype=int)
        hint_counter = 0
        
        for line in sys.stdin:
            # Split the line into parts by tabs
            parts = line.strip().split('\t')
            # Store the corresponding values
            if parts[0] == 'ROW':
                row_hints = [int(x) for x in parts[1:]]
            elif parts[0] == 'COLUMN':
                col_hints = [int(x) for x in parts[1:]]
            elif parts[0] == 'HINT':
                
                row, col, letter = int(parts[1]), int(parts[2]), parts[3]
                board[row][col] = letter
                if letter != "W":
                    row_pieces_placed[row] += 1
                    col_pieces_placed[col] += 1
                    if letter == "C":
                        remaining_pieces["C"] -= 1
                    elif letter == "M":
                        remaining_pieces["M"] -= 1
                    elif letter in ["T", "B", "R", "L"]:
                        remaining_pieces["TBRL"] -= 1
                hint_counter -= 1
                if hint_counter == 0:
                    break
            elif parts[0].isdigit(): 
                hint_counter = int(parts[0])
        return board, row_pieces_placed, col_pieces_placed, remaining_pieces, row_hints, col_hints


    def get_remaining_pieces(self):
        """Retorna o número de peças que ainda faltam colocar no tabuleiro."""
        return sum(self.remaining_pieces.values())


    # A peça C só pode ter ao lado Empty (0) ou Water (W)
    def check_place_C (self, row: int, col: int):
        if self.remaining_pieces["C"] == 0:
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if (above != "" and above != "W") or (below != "" and below != "W"): # se ou encima ou em baixo nao for ou W ou empty -> False
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if (left != "" and left != "W") or (right != "" and right != "W"):
            return False
        return True

    # A peça M só não pode ter peças C ao lado
    def check_place_M (self, row: int, col: int):
        if self.remaining_pieces["M"] == 0:
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above == "C" or below == "C":
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left == "C" or right == "C":
            return False
        return True

    # a peça T so pode ter encima e nos lados ou 0 ou W e em baixo pode ter ou M ou B ou 0
    def check_place_T (self, row: int, col: int):
        if self.remaining_pieces["TBRL"] == 0:
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if (above != "" and above != "W") or below not in ["", "M", "B"]:
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if (left != "" and left != "W") or (right != "" and right != "W"):
            return False
        return True
    # a peça B so pode ter em baixo e nos lados ou 0 ou W e encima pode ter ou M ou T ou 0
    def check_place_B (self, row: int, col: int):
        if self.remaining_pieces["TBRL"] == 0:
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above not in ["", "M", "T"] or (below != "" and below != "W")  :
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if (left != "" and left != "W") or (right != "" and right != "W"):
            return False
        return True

    # a peça R so pode ter à direita ou encima ou em baixo 0 ou W e à esquerda so pode ter M, L ou 0
    def check_place_R (self, row: int, col: int):
        if self.remaining_pieces["TBRL"] == 0:
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if (above != "" and above != "W") or (below != "" and below != "W"):
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left not in ["", "M", "L"] or (right != "" and right != "W"):
            return False
        return True

    # a peça L so pode ter à esquerda ou encima ou em baixo 0 ou W e à direita so pode ter M, R ou 0
    def check_place_L (self, row: int, col: int):
        if self.remaining_pieces["TBRL"] == 0:
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if (above != "" and above != "W") or (below != "" and below != "W"):
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if (left != "" and left != "W") or right not in ["", "M", "R"]:
            return False
        return True
    
    # Option2.
    """Check whether a specific ship can be placed, 
    vertical ships have the given coordinate as the top of the ship
    horizontal ships have the given coordinate as the left of the ship
    they cant exceed the number of pieces in the row or column given by the hints"""
    def check_place_1x1 (self, row: int, col: int):
        if ((self.row_pieces_placed[row] + 1) > self.bimaru.row_hints[row]):
            return False # if adding 1 piece to this row  exceeds the hint value, invalid placement
        if ((self.col_pieces_placed[col] + 1) > self.bimaru.col_hints[col]):
            return False # if adding 1 piece to this column exceeds the hint value, invalid placement
        
        return self.check_place_C(row, col)

    #TODO: ERRO: quando tenta colocar um barco que excede os limites do tabuleiro, dar logo false
    def check_place_1x2_vertical (self, row: int, col: int):
        if row + 1 > 9:
            return False # cannot exceed board limits
        
        if ((self.row_pieces_placed[row] + 1) > self.bimaru.row_hints[row]) or ((self.row_pieces_placed[row + 1] + 1) > self.bimaru.row_hints[row + 1]):
            return False # if adding 1 piece to this row or the one below exceeds the hint value, invalid placement
        if ((self.col_pieces_placed[col] + 2) > self.bimaru.col_hints[col]):
            return False # if adding 2 pieces to this column exceeds the hint value, invalid placement
        
        return self.check_place_T(row,col) and self.check_place_B(row - 1, col)

    def check_place_1x2_horizontal (self, row: int, col: int):
        if col + 1 > 9:
            return False # cannot exceed board limits
        
        if ((self.row_pieces_placed[row] + 2) > self.bimaru.row_hints[row]):
            return False # if adding 2 pieces to this row exceeds the hint value, invalid placement
        if ((self.col_pieces_placed[col] + 1) > self.bimaru.col_hints[col]) or ((self.col_pieces_placed[col + 1] + 1) > self.bimaru.col_hints[col + 1]):
            return False # if adding 1 piece to this column or the next one to the right exceeds the hint value, invalid placement
        
        return self.check_place_L(row,col) and self.check_place_R(row, col +1)

    def check_place_1x3_vertical (self, row: int, col: int):
        if row + 2 > 9:
            return False # cannot exceed board limits
        
        if ((self.row_pieces_placed[row] + 1) > self.bimaru.row_hints[row]) or ((self.row_pieces_placed[row + 1] + 1) > self.bimaru.row_hints[row + 1]) or ((self.row_pieces_placed[row + 2] + 1) > self.bimaru.row_hints[row + 2]):
            return False # if adding 1 piece to this row or the two below exceeds the hint value, invalid placement
        if ((self.col_pieces_placed[col] + 3) > self.bimaru.col_hints[col]):
            return False # if adding 3 pieces to this column exceeds the hint value, invalid placement
        
        return self.check_place_T(row,col) and self.check_place_M(row - 1, col) and self.check_place_B(row - 2, col)

    def check_place_1x3_horizontal (self, row: int, col: int):
        if col + 2 > 9:
            return False # cannot exceed board limits
        
        if ((self.row_pieces_placed[row] + 3) > self.bimaru.row_hints[row]):
            return False # if adding 3 pieces to this row exceeds the hint value, invalid placement
        if ((self.col_pieces_placed[col] + 1) > self.bimaru.col_hints[col]) or ((self.col_pieces_placed[col + 1] + 1) > self.bimaru.col_hints[col + 1]) or ((self.col_pieces_placed[col + 2] + 1) > self.bimaru.col_hints[col + 2]):
            return False # if adding 1 piece to this column or the next two to the right exceeds the hint value, invalid placement
        
        return self.check_place_L(row,col) and self.check_place_M(row, col + 1) and self.check_place_R(row, col + 2)

    def check_place_1x4_vertical (self, row: int, col: int):
        if row + 3 > 9:
            return False # cannot exceed board limits
        
        if ((self.row_pieces_placed[row] + 1) > self.bimaru.row_hints[row]) or ((self.row_pieces_placed[row + 1] + 1) > self.bimaru.row_hints[row + 1]) or ((self.row_pieces_placed[row + 2] + 1) > self.bimaru.row_hints[row + 2]) or ((self.row_pieces_placed[row + 3] + 1) > self.bimaru.row_hints[row + 3]):
            return False # if adding 1 piece to this row or the three below exceeds the hint value, invalid placement
        if ((self.col_pieces_placed[col] + 4) > self.bimaru.col_hints[col]):
            return False # if adding 4 pieces to this column exceeds the hint value, invalid placement
        
        if self.remaining_pieces["M"] < 2: # verificar se temos peças suficientes, porque ao testar M só vai ver se tem 1 duas vezes, e neste caso são precisas duas
            return False
        return self.check_place_T(row,col) and self.check_place_M(row - 1, col) and self.check_place_M(row - 2, col) and self.check_place_B(row - 3, col)

    def check_place_1x4_horizontal (self, row: int, col: int):
        if col + 3 > 9:
            return False # cannot exceed board limits
        
        if ((self.row_pieces_placed[row] + 4) > self.bimaru.row_hints[row]):
            return False # if adding 4 pieces to this row exceeds the hint value, invalid placement
        if ((self.col_pieces_placed[col] + 1) > self.bimaru.col_hints[col]) or ((self.col_pieces_placed[col + 1] + 1) > self.bimaru.col_hints[col + 1]) or ((self.col_pieces_placed[col + 2] + 1) > self.bimaru.col_hints[col + 2]) or ((self.col_pieces_placed[col + 3] + 1) > self.bimaru.col_hints[col + 3]):
            return False # if adding 1 piece to this column or the next three to the right exceeds the hint value, invalid placement
        
        if self.remaining_pieces["M"] < 2: #verificar se temos peças suficientes, porque ao testar M só vai ver se tem 1 duas vezes, e neste caso são precisas duas
            return False
        return self.check_place_L(row,col) and self.check_place_M(row, col + 1) and self.check_place_M(row, col + 2) and self.check_place_R(row, col + 3)

    # insert water around certain pieces
    def insert_water_left (self, row: int, col: int):
        """inserts water to the left of the given position"""
        if 0 <= col - 1 <= 9:
            self.board[row][col - 1] = "W"
    def insert_water_right (self, row: int, col: int):
        """inserts water to the right of the given position"""
        if 0 <= col + 1 <= 9:
            self.board[row][col + 1] = "W"
    def insert_water_below (self, row: int, col: int):
        """inserts water below the given position"""
        if 0 <= row + 1 <= 9:
            self.board[row + 1][col] = "W"
    def insert_water_ontop (self, row: int, col: int):
        """inserts water on top of the given position"""
        if 0 <= row - 1 <= 9:
            self.board[row - 1][col] = "W"
    def insert_water_ontop_below(self, row: int, col: int):
        """Inserts water on top and below the given position"""
        self.insert_water_below(row, col)
        self.insert_water_ontop(row, col)
    def insert_water_right_left(self, row: int, col: int):
        """inserts water to the right and left of the given position"""
        self.insert_water_left(row, col)
        self.insert_water_right(row, col)

    def fill_completed_row_col(self):
        """Fills the rows and cacolums tha aslready have the correct number of pieces"""
        for row in range(10):
            if self.row_pieces_placed[row] == self.bimaru.row_hints[row]: # fill rows that are completed
                for col in range(10):
                    if self.board[row][col] == "":
                        self.board[row][col] = "W"
            if self.col_pieces_placed[row] == self.bimaru.col_hints[row]:
                for col in range(10):
                    if self.board[col][row] == "":
                        self.board[col][row] = "W"
    
    def insert_ship(self, row: int, col: int, piece: str):
        """Inserts a ship at the given position, decreases the pieces count & insert water around piece"""
        if piece == '1x1':
            # place Piece & Water around it
            self.board[row][col] = "C"
            self.insert_water_ontop_below(row, col)
            self.insert_water_right_left(row, col)
            # decrease count of center pieces & increase count of pieces placed
            self.remaining_pieces["C"] -= 1
            self.row_pieces_placed[row] += 1
            self.col_pieces_placed[col] += 1
            
        elif piece == '1x2_vertical':
            # place Top piece & Water around it
            self.board[row][col] = "T"
            self.insert_water_right_left(row, col)
            self.insert_water_ontop(row, col)
            # place Bottom piece & Water around it
            self.board[row + 1][col] = "B"
            self.insert_water_right_left(row + 1, col)
            self.insert_water_below(row + 1, col)
            # decrease count of edge pieces & increase count of pieces placed
            self.remaining_pieces["TBRL"] -= 2
            self.row_pieces_placed[row] += 1
            self.row_pieces_placed[row + 1] += 1
            self.col_pieces_placed[col] += 2
            
        elif piece == '1x2_horizontal':
            # place Left piece & Water around it
            self.board[row][col] = "L"
            self.insert_water_ontop_below(row, col)
            self.insert_water_left(row, col)
            # place Right piece & Water around it
            self.board[row][col + 1] = "R"
            self.insert_water_ontop_below(row, col + 1)
            self.insert_water_right(row, col + 1)
            # decrease count of edge pieces & increase count of pieces placed
            self.remaining_pieces["TBRL"] -= 2
            self.row_pieces_placed[row] += 2
            self.col_pieces_placed[col] += 1
            self.col_pieces_placed[col + 1] += 1
            
        elif piece == '1x3_vertical':
            # place Top piece & Water around it
            self.board[row][col] = "T"
            self.insert_water_right_left(row, col)
            self.insert_water_ontop(row, col)
            # place Middle piece & Water around it
            self.board[row + 1][col] = "M"
            self.insert_water_right_left(row + 1, col)
            # place Bottom piece & Water around it
            self.board[row + 2][col] = "B"
            self.insert_water_right_left(row + 2, col)
            self.insert_water_below(row + 2, col)
            # decrease count of edge & middle pieces & increase count of pieces placed
            self.remaining_pieces["TBRL"] -= 2
            self.remaining_pieces["M"] -= 1
            self.row_pieces_placed[row] += 1
            self.row_pieces_placed[row + 1] += 1
            self.row_pieces_placed[row + 2] += 1
            self.col_pieces_placed[col] += 3
            
        elif piece == '1x3_horizontal':
            # place Left piece & Water around it
            self.board[row][col] = "L"
            self.insert_water_ontop_below(row, col)
            self.insert_water_left(row, col)
            # place Middle piece & Water around it
            self.board[row][col + 1] = "M"
            self.insert_water_ontop_below(row, col + 1)
            # place Right piece & Water around it
            self.board[row][col + 2] = "R"
            self.insert_water_ontop_below(row, col + 2)
            self.insert_water_right(row, col + 2)
            # decrease count of edge & middle pieces & increase count of pieces placed
            self.remaining_pieces["TBRL"] -= 2
            self.remaining_pieces["M"] -= 1
            self.row_pieces_placed[row] += 3
            self.col_pieces_placed[col] += 1
            self.col_pieces_placed[col + 1] += 1
            self.col_pieces_placed[col + 2] += 1
            
        elif piece == '1x4_vertical':
            # place Top piece & Water around it
            self.board[row][col] = "T"
            self.insert_water_right_left(row, col)
            self.insert_water_ontop(row, col)
            # place Middle piece & Water around it
            self.board[row + 1][col] = "M"
            self.insert_water_right_left(row + 1, col)
            # place Middle piece & Water around it
            self.board[row + 2][col] = "M"
            self.insert_water_right_left(row + 2, col)
            # place Bottom piece & Water around it
            self.board[row + 3][col] = "B"
            self.insert_water_right_left(row + 3, col)
            self.insert_water_below(row + 3, col)
            # decrease count of edge & middle pieces & increase count of pieces placed
            self.remaining_pieces["TBRL"] -= 2
            self.remaining_pieces["M"] -= 2
            self.row_pieces_placed[row] += 1
            self.row_pieces_placed[row + 1] += 1
            self.row_pieces_placed[row + 2] += 1
            self.row_pieces_placed[row + 3] += 1
            self.col_pieces_placed[col] += 4
            
        elif piece == '1x4_horizontal':
            # place Left piece & Water around it
            self.board[row][col] = "L"
            self.insert_water_ontop_below(row, col)
            self.insert_water_left(row, col)
            # place Middle piece & Water around it
            self.board[row][col + 1] = "M"
            self.insert_water_ontop_below(row, col + 1)
            # place Middle piece & Water around it
            self.board[row][col + 2] = "M"
            self.insert_water_left(row, col + 2)
            #place Right piece & Water around it
            self.board[row][col + 3] = "R"
            self.insert_water_ontop_below(row, col + 3)
            self.insert_water_right(row, col + 3)
            # decrease count of edge & middle pieces & increase count of pieces placed
            self.remaining_pieces["TBRL"] -= 2
            self.remaining_pieces["M"] -= 2
            self.row_pieces_placed[row] += 4
            self.col_pieces_placed[col] += 1
            self.col_pieces_placed[col + 1] += 1
            self.col_pieces_placed[col + 2] += 1
            self.col_pieces_placed[col + 3] += 1
        
        self.fill_completed_row_col() # Fill rows and columns that are completed with water


class Bimaru(Problem):
    def __init__(self, board, row_pieces_placed, col_pieces_placed, remaining_pieces, row_hints, col_hints):
        """O construtor especifica o estado inicial."""
        # number of positions in the row / column with a ship cell
        self.row_hints = row_hints 
        self.col_hints = col_hints
        board_object = Board(board, row_pieces_placed, col_pieces_placed, remaining_pieces, self) #Criar o Board inicial, passando o problema Bimaru para poder aceder às hints
        self.state = BimaruState(board_object)
        super().__init__(self.state)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        for row in range(10):
            for col in range(10):
                if state.board.get_remaining_pieces() == 0: # Just precaution, if all pieces are placed
                    break
                if state.board.get_value(row, col) == "": # Only tries to place a piece in an empty cell
                        #Two possible approaches:
                            # 1. Try to Place an Indicidual Piece: C, M, T, B, R, L
                            # 2. Try to place a Ship (Horizontal and Vertical): 1x1, 1x2, 1x3, 1x4 (Centered on the topmost/left most piece)
                        # Option 2.
                        # try to plae a 1x1 ship (on current cell)
                        if state.board.check_place_1x1(row,col):
                            actions.append((row, col, "1x1"))
                        # try to plae a 1x2 ship vertivaly (topmost square on current cell)
                        if state.board.check_place_1x2_vertical(row,col):
                            actions.append((row, col, "1x2_vertical"))
                        # try to plae a 1x2 ship horizontaly (leftmost square on current cell)
                        if state.board.check_place_1x2_horizontal(row,col):
                            actions.append((row, col, "1x2_horizontal"))
                        # try to plae a 1x3 ship vertivaly (topmost square on current cell)
                        if state.board.check_place_1x3_vertical(row,col):
                            actions.append((row, col, "1x3_vertical"))
                        # try to plae a 1x3 ship horizontaly (leftmost square on current cell)
                        if state.board.check_place_1x3_horizontal(row,col):
                            actions.append((row, col, "1x3_horizontal"))
                        # try to plae a 1x4 ship vertivaly (topmost square on current cell)
                        if state.board.check_place_1x4_vertical(row,col):
                            actions.append((row, col, "1x4_vertical"))
                        # try to plae a 1x4 ship horizontaly (leftmost square on current cell)
                        if state.board.check_place_1x4_horizontal(row,col):
                            actions.append((row, col, "1x4_horizontal"))

        return actions 

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        row, col, ship = action
        new_board = copy.deepcopy(state.board)
        new_state = BimaruState(new_board)
        new_state.board.insert_ship(row, col, ship)
        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        ######DEBUG############TODO
        print(state.board.board)
        print("\n\n")
        ######DEBUG############
        return state.board.get_remaining_pieces() == 0

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO: possible additions
            # numero de celulas vazias -> menos é melhor
        print(node.state.board.get_remaining_pieces()) # DEBUG TOD
        return node.state.board.get_remaining_pieces()


if __name__ == "__main__":
    # Ler o ficheiro do standard input,
    board, row_pieces_placed, col_pieces_placed, remaining_pieces, row_hints, col_hints = Board.parse_instance()
    # Criar uma instância do problema Bimaru,
    problem = Bimaru(board, row_pieces_placed, col_pieces_placed, remaining_pieces, row_hints, col_hints)
    print(problem.state.board.board)
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    goal_node = astar_search(problem)
    # Imprimir para o standard output no formato indicado.
    # TODO
    # Replace 'W' with '.'
    # solved_board = np.where(goal_node.state.board  == 'W', '.', goal_node.state.board )
    # print(solved_board)
