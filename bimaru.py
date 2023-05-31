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
    
    def __init__(self, board, remaining_pieces, unfinished_hints, remaining_ships, bimaru):
        self.board = board
        self.bimaru = bimaru # the bimaru problem object, to access the rows & columns hints
        self.remaining_pieces = remaining_pieces # total number of pieces to be placed
        self.unfinished_hints = unfinished_hints
        self.remaining_ships = remaining_ships
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
    
    def adjacent_diagonal_values(self, row: int, col: int) -> Tuple[str, str, str, str]:
        """Devolve os valores imediatamente acima, abaixo, à esquerda e à direita, respectivamente."""
        if(row > 0 and col > 0):
            above_left = self.get_value(row-1, col-1) 
        else:
            above_left = ""
        
        if(row > 0 and col < 9):
            above_right = self.get_value(row-1, col+1)
        else:
            above_right = ""
        
        if(row < 9 and col > 0):
            below_left = self.get_value(row+1, col-1)
        else:
            below_left = ""
        
        if(row < 9 and col < 9):
            below_right = self.get_value(row+1, col+1)
        else:
            below_right = ""

        return above_left, above_right, below_left, below_right
        
    def row_pieces_placed (self, row_index: int) -> int:
        """Devolve o número de peças colocadas numa linha."""
        row = self.board[row_index]
        return np.count_nonzero((row == "T") | (row == "B") | (row == "L") | (row == "R") | (row == "C") | (row == "M"))

    def col_pieces_placed(self, col_index: int) -> int:
        """Return the number of pieces placed in a column."""
        column = self.board[:, col_index]
        return np.count_nonzero((column == "T") | (column == "B") | (column == "L") | (column == "R") | (column == "C") | (column == "M"))
    
    def get_smallest_row_col_remaining_pieces(self):
        """for heuristic, returns the smalllest non zero number of remaining pieces remaining on a non completed row / column """
        lowest = abs(self.row_pieces_placed(0) - self.bimaru.row_hints[0])
        for index in range(10):
            diff = abs(self.row_pieces_placed(index) - self.bimaru.row_hints[index])
            if diff < lowest and diff != 0:
                lowest = diff
        for index in range(10):
            diff = abs(self.col_pieces_placed(index) - self.bimaru.col_hints[index])
            if diff < lowest and diff != 0:
                lowest = diff
        return lowest
    
    def fill_water_around_hints(self):
        for i in range(10):
            row = i
            for j in range(10):
                col = j
                if board[i][j] == "C":
                    self.insert_water_ontop_below(row, col)
                    self.insert_water_right_left(row, col)
                    self.insert_water_diagonals(row, col)
                elif board[i][j] == "M":
                    self.insert_water_diagonals(row, col)
                elif board[i][j] in ["T", "B", "R", "L"]:
                    if board[i][j] == "T":
                        self.insert_water_right_left(row, col)
                        self.insert_water_ontop(row, col)
                        self.insert_water_diagonals(row, col)
                    elif board[i][j] == "B":
                        self.insert_water_right_left(row, col)
                        self.insert_water_below(row, col)
                        self.insert_water_diagonals(row, col)
                    elif board[i][j] == "R":
                        self.insert_water_ontop_below(row, col)
                        self.insert_water_right(row, col)
                        self.insert_water_diagonals(row, col)
                    elif board[i][j] == "L":
                        self.insert_water_ontop_below(row, col)
                        self.insert_water_left(row, col)
                        self.insert_water_diagonals(row, col)

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        row_hints = []
        col_hints = []
        remaining_pieces = {"C": 4, "M": 4, "TBRL": 12} # initial number of pieces to be placed on the board
        remaining_ships = {"1x1": 4, "1x2": 3, "1x3": 2 , "1x4": 1}
        
        board = np.zeros((10, 10), dtype=str)
        hint_counter = 0
        unfinished_hints = []
        
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
                    if letter == "C":
                        remaining_pieces["C"] -= 1
                        remaining_ships["1x1"] -= 1
                    elif letter == "M":
                        unfinished_hints.append((row, col))
                        remaining_pieces["M"] -= 1
                    elif letter in ["T", "B", "R", "L"]:
                        unfinished_hints.append((row, col))
                        remaining_pieces["TBRL"] -= 1

                hint_counter -= 1
                if hint_counter == 0:
                    break
            elif parts[0].isdigit(): 
                hint_counter = int(parts[0])
        return board, remaining_pieces, row_hints, col_hints, unfinished_hints, remaining_ships


    def get_remaining_pieces(self):
        """Retorna o número de peças que ainda faltam colocar no tabuleiro."""
        return sum(self.remaining_pieces.values())

    """Check if specific piece can be placed in a specific position"""
    
    # A peça C só pode ter ao lado Empty (0) ou Water (W)
    def check_place_C (self, row: int, col: int):
        if self.remaining_pieces["C"] == 0:
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above not in ["", "W"] or below not in ["", "W"]:
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left not in ["", "W"] or right not in ["", "W"]:
            return False
        above_left, above_right, below_left, below_right = self.adjacent_diagonal_values(row, col)
        if above_left not in ["", "W"] or above_right not in ["", "W"] or below_left not in ["", "W"] or below_right not in ["", "W"]:
            return False
        
        return True

    # A peça M vertical só pode ter peças W ou empty de lado ou Top/M Encima e Bottom/M Embaixo
    def check_place_M_vertical (self, row: int, col: int):
        if self.board[row][col] != "M" and self.remaining_pieces["M"] == 0: # Only needs to check if there are available pieces if the pice is not already placed (as a hint)
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above not in ["T", "M", ""] or below not in ["B", "M", ""]:
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left not in ["", "W"] or right not in ["", "W"]:
            return False
        above_left, above_right, below_left, below_right = self.adjacent_diagonal_values(row, col)
        if above_left not in ["", "W"] or above_right not in ["", "W"] or below_left not in ["", "W"] or below_right not in ["", "W"]:
            return False
        
        return True
    
    # A peça M horizontal só pode ter peças W ou empty encima ou Left/M Esquerda e Right/M Direita
    def check_place_M_horizontal (self, row: int, col: int):
        if self.board[row][col] != "M" and self.remaining_pieces["M"] == 0: # Only needs to check if there are available pieces if the pice is not already placed (as a hint)
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above not in ["", "W"] or below not in ["", "W"]:
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left not in ["L", "M", ""] or right not in ["R", "M", ""]:
            return False
        above_left, above_right, below_left, below_right = self.adjacent_diagonal_values(row, col)
        if above_left not in ["", "W"] or above_right not in ["", "W"] or below_left not in ["", "W"] or below_right not in ["", "W"]:
            return False
        
        return True

    # a peça T so pode ter encima e nos lados ou 0 ou W e em baixo pode ter ou M ou B ou 0
    def check_place_T (self, row: int, col: int):
        if self.board[row][col] != "TLBR" and self.remaining_pieces["TBRL"] == 0: # Only needs to check if there are available pieces if the pice is not already placed (as a hint)
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above not in ["", "W"] or below not in ["", "M", "B"]:
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left not in ["", "W"] or right not in ["", "W"]:
            return False
        above_left, above_right, below_left, below_right = self.adjacent_diagonal_values(row, col)
        if above_left not in ["", "W"] or above_right not in ["", "W"] or below_left not in ["", "W"] or below_right not in ["", "W"]:
            return False
        
        return True
    # a peça B so pode ter em baixo e nos lados ou 0 ou W e encima pode ter ou M ou T ou 0
    def check_place_B (self, row: int, col: int):
        if self.board[row][col] != "TLBR" and self.remaining_pieces["TBRL"] == 0: # Only needs to check if there are available pieces if the pice is not already placed (as a hint)
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above not in ["", "M", "T"] or below not in ["", "W"]:
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left not in ["", "W"] or right not in ["", "W"]:
            return False
        above_left, above_right, below_left, below_right = self.adjacent_diagonal_values(row, col)
        if above_left not in ["", "W"] or above_right not in ["", "W"] or below_left not in ["", "W"] or below_right not in ["", "W"]:
            return False
        
        return True

    # a peça R so pode ter à direita ou encima ou em baixo 0 ou W e à esquerda so pode ter M, L ou 0
    def check_place_R (self, row: int, col: int):
        if self.board[row][col] != "TLBR" and self.remaining_pieces["TBRL"] == 0: # Only needs to check if there are available pieces if the pice is not already placed (as a hint)
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above not in ["", "W"] or below not in ["", "W"]:
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left not in ["", "M", "L"] or right not in ["", "W"]:
            return False
        above_left, above_right, below_left, below_right = self.adjacent_diagonal_values(row, col)
        if above_left not in ["", "W"] or above_right not in ["", "W"] or below_left not in ["", "W"] or below_right not in ["", "W"]:
            return False
        
        return True

    # a peça L so pode ter à esquerda ou encima ou em baixo 0 ou W e à direita so pode ter M, R ou 0
    def check_place_L (self, row: int, col: int):
        if self.board[row][col] != "TLBR" and self.remaining_pieces["TBRL"] == 0: # Only needs to check if there are available pieces if the pice is not already placed (as a hint)
            return False
        above, below = self.adjacent_vertical_values(row, col)
        if above not in ["", "W"] or below not in ["", "W"]:
            return False
        left, right = self.adjacent_horizontal_values(row, col)
        if left not in ["", "W"] or right not in ["", "M", "R"]:
            return False
        above_left, above_right, below_left, below_right = self.adjacent_diagonal_values(row, col)
        if above_left not in ["", "W"] or above_right not in ["", "W"] or below_left not in ["", "W"] or below_right not in ["", "W"]:
            return False
        
        return True

    """
    Check whether a specific ship can be placed in a given coordinate: 
    vertical ships have the given coordinate as the top of the ship
    horizontal ships have the given coordinate as the left of the ship
    they cant exceed the number of pieces in the row or column given by the hints, among other restrictions
    """
    def check_place_1x1 (self, row: int, col: int):
        if self.remaining_ships["1x1"] == 0:
            return False
        if ((self.row_pieces_placed(row) + 1) > self.bimaru.row_hints[row]): # adding a C will also add water to the sorouding cells
            return False # if adding 1 piece to this row  exceeds the hint value, invalid placement
        if ((self.col_pieces_placed(col) + 1) > self.bimaru.col_hints[col]):
            return False # if adding 1 piece to this column exceeds the hint value, invalid placement
        
        return self.check_place_C(row, col)

    def check_place_1x2_vertical (self, row: int, col: int):
        if self.remaining_ships["1x2"] == 0:
            return False
        
        if row < 0 or row + 1 > 9:
            return False # cannot exceed board limits
        
        # Check if number of pieces in the column will exceed limit
        pieces_to_be_placed_col = 0
        if self.board[row][col] == "":
            pieces_to_be_placed_col += 1
        if self.board[row + 1][col] == "":
            pieces_to_be_placed_col += 1
        if pieces_to_be_placed_col + self.col_pieces_placed(col)  > self.bimaru.col_hints[col]: # if the number of pieces that need to be added to place this ship + the ones that are already placed on the column exceed the column limit -> invalid placement
            return False
        
        # Check if number of pieces in each row will exceed limit
        if self.board[row][col] == "" and ((self.row_pieces_placed(row) + 1) > self.bimaru.row_hints[row]):
            return False
        if self.board[row + 1][col] == "" and ((self.row_pieces_placed(row + 1) + 1) > self.bimaru.row_hints[row + 1]):
            return False
        
        return self.check_place_T(row,col) and self.check_place_B(row + 1, col)

    def check_place_1x2_horizontal (self, row: int, col: int):
        if self.remaining_ships["1x2"] == 0:
            return False
        
        if col < 0 or col + 1 > 9:
            return False # cannot exceed board limits
        
        # Check if number of pieces in the row will exceed limit
        pieces_to_be_placed_row = 0
        if self.board[row][col] == "":
            pieces_to_be_placed_row += 1
        if self.board[row][col + 1] == "":
            pieces_to_be_placed_row += 1
        if pieces_to_be_placed_row + self.row_pieces_placed(row)  > self.bimaru.row_hints[row]: # if the number of pieces that need to be added to place this ship + the ones that are already placed on the row exceed the column limit -> invalid placement
            return False
        
        # Check if number of pieces in each column will exceed limit
        if self.board[row][col] == "" and ((self.col_pieces_placed(col) + 1) > self.bimaru.col_hints[col]):
            return False
        if self.board[row][col + 1] == "" and ((self.col_pieces_placed(col + 1) + 1) > self.bimaru.col_hints[col + 1]):
            return False
        
        return self.check_place_L(row,col) and self.check_place_R(row, col +1)

    def check_place_1x3_vertical (self, row: int, col: int):
        if self.remaining_ships["1x3"] == 0:
            return False
        
        if row < 0 or row + 2 > 9:
            return False # cannot exceed board limits
        
        # Check if number of pieces in the column will exceed limit
        pieces_to_be_placed_col = 0
        if self.board[row][col] == "":
            pieces_to_be_placed_col += 1
        if self.board[row + 1][col] == "":
            pieces_to_be_placed_col += 1
        if self.board[row + 2][col] == "":
            pieces_to_be_placed_col += 1
        if pieces_to_be_placed_col + self.col_pieces_placed(col)  > self.bimaru.col_hints[col]: # if the number of pieces that need to be added to place this ship + the ones that are already placed on the column exceed the column limit -> invalid placement
            return False
        
        # Check if number of pieces in each row will exceed limit
        if self.board[row][col] == "" and ((self.row_pieces_placed(row) + 1) > self.bimaru.row_hints[row]):
            return False
        if self.board[row + 1][col] == "" and ((self.row_pieces_placed(row + 1) + 1) > self.bimaru.row_hints[row + 1]):
            return False
        if self.board[row + 2][col] == "" and ((self.row_pieces_placed(row + 2) + 1) > self.bimaru.row_hints[row + 2]):
            return False
        
        return self.check_place_T(row,col) and self.check_place_M_vertical(row + 1, col) and self.check_place_B(row + 2, col)

    def check_place_1x3_horizontal (self, row: int, col: int):
        if self.remaining_ships["1x3"] == 0:
            return False
        
        if col < 0 or col + 2 > 9:
            return False # cannot exceed board limits
        
        
        # Check if number of pieces in the row will exceed limit
        pieces_to_be_placed_row = 0
        if self.board[row][col] == "":
            pieces_to_be_placed_row += 1
        if self.board[row][col + 1] == "":
            pieces_to_be_placed_row += 1
        if self.board[row][col + 2] == "":
            pieces_to_be_placed_row += 1
        if pieces_to_be_placed_row + self.row_pieces_placed(row)  > self.bimaru.row_hints[row]: # if the number of pieces that need to be added to place this ship + the ones that are already placed on the row exceed the column limit -> invalid placement
            return False
        
        # Check if number of pieces in each column will exceed limit
        if self.board[row][col] == "" and ((self.col_pieces_placed(col) + 1) > self.bimaru.col_hints[col]):
            return False
        if self.board[row][col + 1] == "" and ((self.col_pieces_placed(col + 1) + 1) > self.bimaru.col_hints[col + 1]):
            return False
        if self.board[row][col + 2] == "" and ((self.col_pieces_placed(col + 2) + 1) > self.bimaru.col_hints[col + 2]):
            return False
        
        return self.check_place_L(row,col) and self.check_place_M_horizontal(row, col + 1) and self.check_place_R(row, col + 2)

    def check_place_1x4_vertical (self, row: int, col: int):
        if self.remaining_ships["1x4"] == 0:
            return False
        
        if row < 0 or row + 3 > 9:
            return False # cannot exceed board limits
        
        # Check if number of pieces in the column will exceed limit
        pieces_to_be_placed_col = 0
        if self.board[row][col] == "":
            pieces_to_be_placed_col += 1
        if self.board[row + 1][col] == "":
            pieces_to_be_placed_col += 1
        if self.board[row + 2][col] == "":
            pieces_to_be_placed_col += 1
        if self.board[row + 3][col] == "":
            pieces_to_be_placed_col += 1
        if pieces_to_be_placed_col + self.col_pieces_placed(col)  > self.bimaru.col_hints[col]: # if the number of pieces that need to be added to place this ship + the ones that are already placed on the column exceed the column limit -> invalid placement
            return False
        
        # Check if number of pieces in each row will exceed limit
        if self.board[row][col] == "" and ((self.row_pieces_placed(row) + 1) > self.bimaru.row_hints[row]):
            return False
        if self.board[row + 1][col] == "" and ((self.row_pieces_placed(row + 1) + 1) > self.bimaru.row_hints[row + 1]):
            return False
        if self.board[row + 2][col] == "" and ((self.row_pieces_placed(row + 2) + 1) > self.bimaru.row_hints[row + 2]):
            return False
        if self.board[row + 3][col] == "" and ((self.row_pieces_placed(row + 3) + 1) > self.bimaru.row_hints[row + 3]):
            return False
        
        # check if there are enough M pieces to be placed if none are alreday placed, (because check place M will only check if there is one piece available 2 times)
        M1 = self.board[row + 1][col]
        M2 = self.board[row + 2][col]
        if M1 != "M" and M2 != "M" and self.remaining_pieces["M"] < 2:
            return False
        if M1 != "M" and M2 == "M" and self.remaining_pieces["M"] < 1:
            return False
        if M1 == "M" and M2 != "M" and self.remaining_pieces["M"] < 1:
            return False
        
        return self.check_place_T(row,col) and self.check_place_M_vertical(row + 1, col) and self.check_place_M_vertical(row + 2, col) and self.check_place_B(row + 3, col)

    def check_place_1x4_horizontal (self, row: int, col: int):
        if self.remaining_ships["1x4"] == 0:
            return False
        
        if col < 0 or  col + 3 > 9:
            return False # cannot exceed board limits
        
        # Check if number of pieces in the row will exceed limit
        pieces_to_be_placed_row = 0
        if self.board[row][col] == "":
            pieces_to_be_placed_row += 1
        if self.board[row][col + 1] == "":
            pieces_to_be_placed_row += 1
        if self.board[row][col + 2] == "":
            pieces_to_be_placed_row += 1
        if self.board[row][col + 3] == "":
            pieces_to_be_placed_row += 1
        if pieces_to_be_placed_row + self.row_pieces_placed(row)  > self.bimaru.row_hints[row]: # if the number of pieces that need to be added to place this ship + the ones that are already placed on the row exceed the column limit -> invalid placement
            return False
        
        # Check if number of pieces in each column will exceed limit
        if self.board[row][col] == "" and ((self.col_pieces_placed(col) + 1) > self.bimaru.col_hints[col]):
            return False
        if self.board[row][col + 1] == "" and ((self.col_pieces_placed(col + 1) + 1) > self.bimaru.col_hints[col + 1]):
            return False
        if self.board[row][col + 2] == "" and ((self.col_pieces_placed(col + 2) + 1) > self.bimaru.col_hints[col + 2]):
            return False
        if self.board[row][col + 3] == "" and ((self.col_pieces_placed(col + 3) + 1) > self.bimaru.col_hints[col + 3]):
            return False
        
        # check if there are enough M pieces to be placed if none are alreday placed, (because check place M will only check if there is one piece available 2 times)
        M1 = self.board[row][col + 1]
        M2 = self.board[row][col + 2]
        if M1 != "M" and M2 != "M" and self.remaining_pieces["M"] < 2:
            return False
        if M1 != "M" and M2 == "M" and self.remaining_pieces["M"] < 1:
            return False
        if M1 == "M" and M2 != "M" and self.remaining_pieces["M"] < 1:
            return False
        
        return self.check_place_L(row,col) and self.check_place_M_horizontal(row, col + 1) and self.check_place_M_horizontal(row, col + 2) and self.check_place_R(row, col + 3)

    """Used to check if an action is already in the list, ignoring the last 2 fields"""
    def tuple_doesnt_exist(self, list, new_tuple):
        for tuple in list:
            if tuple[:4] == new_tuple[:4]:
                return False #Exists already
        return True
    
    def hint_actions (self):
        actions = []
        hints_to_remove = []
        
        for hint in self.unfinished_hints:
            row, col = hint
            if self.get_value(row, col) in ["T", "B", "L", "R", "M"]:
                if self.board[row][col] == "T": # Try valid Ship placements around an T piece
                    if self.remaining_ships["1x4"] > 0:
                        if self.check_place_1x4_vertical(row,col):
                            new_action = (row, col, "1x4_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x3"] > 0:
                        if self.check_place_1x3_vertical(row,col):
                            new_action = (row, col, "1x3_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x2"] > 0:
                        if self.check_place_1x2_vertical(row,col):
                            new_action = (row, col, "1x2_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                
                elif self.board[row][col] == "L": # Try valid Ship placements around an L piece
                    if self.remaining_ships["1x4"] > 0:
                        if self.check_place_1x4_horizontal(row,col):
                            new_action = (row, col, "1x4_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x3"] > 0:
                        if self.check_place_1x3_horizontal(row,col):
                            new_action = (row, col, "1x3_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x2"] > 0:
                        if self.check_place_1x2_horizontal(row,col):
                            new_action = (row, col, "1x2_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                
                elif self.board[row][col] == "B": # Try valid Ship placements around a B piece
                    if self.remaining_ships["1x4"] > 0:
                        if self.check_place_1x4_vertical(row - 3,col):
                            new_action = (row - 3, col, "1x4_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x3"] > 0:
                        if self.check_place_1x3_vertical(row - 2,col):
                            new_action = (row - 2, col, "1x3_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x2"] > 0:
                        if self.check_place_1x2_vertical(row - 1,col):
                            new_action = (row - 1, col, "1x2_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                
                elif self.board[row][col] == "R": # Try valid Ship placements around a R piece
                    if self.remaining_ships["1x4"] > 0:
                        if self.check_place_1x4_horizontal(row,col - 3):
                            new_action = (row, col - 3, "1x4_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x3"] > 0:
                        if self.check_place_1x3_horizontal(row,col - 2):
                            new_action = (row, col - 2, "1x3_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x2"] > 0:
                        if self.check_place_1x2_horizontal(row,col - 1):
                            new_action = (row, col - 1, "1x2_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                
                elif self.board[row][col] == "M": # Try valid Ship placements around a M piece
                    if self.remaining_ships["1x4"] > 0:
                        if self.check_place_1x4_vertical(row - 1,col):
                            new_action = (row - 1, col, "1x4_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                        if self.check_place_1x4_vertical(row - 2,col):
                            new_action = (row - 2, col, "1x4_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                        
                        if self.check_place_1x4_horizontal(row,col - 2):
                            new_action = (row, col - 2, "1x4_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                        if self.check_place_1x4_horizontal(row,col - 1):
                            new_action = (row, col - 1, "1x4_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    if self.remaining_ships["1x3"] > 0:
                        if self.check_place_1x3_vertical(row - 1,col):
                            new_action = (row - 1, col, "1x3_vertical", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                        if self.check_place_1x3_horizontal(row,col - 1):
                            new_action = (row, col - 1, "1x3_horizontal", "hint", row, col)
                            if self.tuple_doesnt_exist(actions, new_action):
                                actions.append(new_action)
                            else:
                                hints_to_remove.append((row, col))
                    
        for hint in hints_to_remove:
            self.unfinished_hints.remove(hint)
        return actions
    

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
    
    def insert_water_diagonals (self, row: int, col: int):
        """inserts water diagonally around the given position"""
        self.insert_water_top_left_diagonal(row, col)
        self.insert_water_top_right_diagonal(row, col)
        self.insert_water_below_left_diagonal(row, col)
        self.insert_water_below_right_diagonal(row, col)

    def insert_water_top_right_diagonal (self, row: int, col: int):
        """Inserts water on top and to the right of the given position"""
        if 0 <= row - 1 <= 9 and 0 <= col + 1 <= 9:
            self.board[row - 1][col + 1] = "W"
    
    def insert_water_top_left_diagonal (self, row: int, col: int):
        """Inserts water on top and to the left of the given position"""
        if 0 <= row - 1 <= 9 and 0 <= col - 1 <= 9:
            self.board[row - 1][col - 1] = "W"

    def insert_water_below_right_diagonal (self, row: int, col: int):
        """Inserts water below and to the right of the given position"""
        if 0 <= row + 1 <= 9 and 0 <= col + 1 <= 9:
            self.board[row + 1][col + 1] = "W"
        
    def insert_water_below_left_diagonal (self, row: int, col: int):
        """Inserts water below and to the left of the given position"""
        if 0 <= row + 1 <= 9 and 0 <= col - 1 <= 9:
            self.board[row + 1][col - 1] = "W"

    def insert_water_ontop_below(self, row: int, col: int):
        """Inserts water on top and below the given position"""
        self.insert_water_below(row, col)
        self.insert_water_ontop(row, col)
    def insert_water_right_left(self, row: int, col: int):
        """inserts water to the right and left of the given position"""
        self.insert_water_left(row, col)
        self.insert_water_right(row, col)

    def fill_completed_row_col(self):
        """Fills the rows and colums that already have the correct number of pieces"""
        for index in range(10):
            if self.row_pieces_placed(index) == self.bimaru.row_hints[index]: # fill rows that are completed
                for col in range(10):
                    if self.board[index][col] == "":
                        self.board[index][col] = "W"
            if self.col_pieces_placed(index) == self.bimaru.col_hints[index]:
                for row in range(10):
                    if self.board[row][index] == "":
                        self.board[row][index] = "W"
    
    def insert_ship(self, row: int, col: int, piece: str):
        """Inserts a ship at the given position, decreases the pieces count & insert water around piece"""
        if piece == '1x1':
            # place Piece & Water around it
            self.board[row][col] = "C"
            self.insert_water_ontop_below(row, col)
            self.insert_water_right_left(row, col)
            self.insert_water_diagonals(row, col)
            # decrease count of center pieces
            self.remaining_ships["1x1"] -= 1
            self.remaining_pieces["C"] -= 1
            
        elif piece == '1x2_vertical':
            # place Top piece & Water around it
            if self.board[row][col] == "T":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col] = "T"
            self.insert_water_right_left(row, col)
            self.insert_water_ontop(row, col)
            self.insert_water_diagonals(row, col)
            # place Bottom piece & Water around it
            if self.board[row + 1][col] == "B":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row + 1][col] = "B"
            self.insert_water_right_left(row + 1, col)
            self.insert_water_below(row + 1, col)
            self.insert_water_diagonals(row, col)
            # decrease count of edge pieces
            self.remaining_ships["1x2"] -= 1
            self.remaining_pieces["TBRL"] -= 2

            
        elif piece == '1x2_horizontal':
            # place Left piece & Water around it
            if self.board[row][col] == "L":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col] = "L"
            self.insert_water_ontop_below(row, col)
            self.insert_water_left(row, col)
            self.insert_water_diagonals(row, col)
            # place Right piece & Water around it
            if self.board[row][col + 1] == "R":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col + 1] = "R"
            self.insert_water_ontop_below(row, col + 1)
            self.insert_water_right(row, col + 1)
            self.insert_water_diagonals(row, col)
            # decrease count of edge pieces
            self.remaining_ships["1x2"] -= 1
            self.remaining_pieces["TBRL"] -= 2
            
        elif piece == '1x3_vertical':
            # place Top piece & Water around it
            if self.board[row][col] == "T":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col] = "T"
            self.insert_water_right_left(row, col)
            self.insert_water_ontop(row, col)
            self.insert_water_diagonals(row, col)
            # place Middle piece & Water around it
            if self.board[row + 1][col] == "M":
                self.remaining_pieces["M"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row + 1][col] = "M"
            self.insert_water_right_left(row + 1, col)
            self.insert_water_diagonals(row, col)
            # place Bottom piece & Water around it
            if self.board[row + 2][col] == "B":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row + 2][col] = "B"
            self.insert_water_right_left(row + 2, col)
            self.insert_water_below(row + 2, col)
            self.insert_water_diagonals(row, col)
            # decrease count of edge & middle pieces
            self.remaining_ships["1x3"] -= 1
            self.remaining_pieces["TBRL"] -= 2
            self.remaining_pieces["M"] -= 1
            
        elif piece == '1x3_horizontal':
            # place Left piece & Water around it
            if self.board[row][col] == "L":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col] = "L"
            self.insert_water_ontop_below(row, col)
            self.insert_water_left(row, col)
            self.insert_water_diagonals(row, col)
            # place Middle piece & Water around it
            if self.board[row][col + 1] == "M":
                self.remaining_pieces["M"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col + 1] = "M"
            self.insert_water_ontop_below(row, col + 1)
            self.insert_water_diagonals(row, col)
            # place Right piece & Water around it
            if self.board[row][col + 2] == "R":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col + 2] = "R"
            self.insert_water_ontop_below(row, col + 2)
            self.insert_water_right(row, col + 2)
            self.insert_water_diagonals(row, col)
            # decrease count of edge & middle pieces
            self.remaining_ships["1x3"] -= 1
            self.remaining_pieces["TBRL"] -= 2
            self.remaining_pieces["M"] -= 1
            
        elif piece == '1x4_vertical':
            # place Top piece & Water around it
            if self.board[row][col] == "T":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col] = "T"
            self.insert_water_right_left(row, col)
            self.insert_water_ontop(row, col)
            self.insert_water_diagonals(row, col)
            # place Middle piece & Water around it
            if self.board[row + 1][col] == "M":
                self.remaining_pieces["M"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row + 1][col] = "M"
            self.insert_water_right_left(row + 1, col)
            self.insert_water_diagonals(row, col)
            # place Middle piece & Water around it
            if self.board[row + 2][col] == "M":
                self.remaining_pieces["M"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row + 2][col] = "M"
            self.insert_water_right_left(row + 2, col)
            self.insert_water_diagonals(row, col)
            # place Bottom piece & Water around it
            if self.board[row + 3][col] == "B":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row + 3][col] = "B"
            self.insert_water_right_left(row + 3, col)
            self.insert_water_below(row + 3, col)
            self.insert_water_diagonals(row, col)
            # decrease count of edge & middle pieces
            self.remaining_ships["1x4"] -= 1
            self.remaining_pieces["TBRL"] -= 2
            self.remaining_pieces["M"] -= 2
            
        elif piece == '1x4_horizontal':
            # place Left piece & Water around it
            if self.board[row][col] == "L":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col] = "L"
            self.insert_water_ontop_below(row, col)
            self.insert_water_left(row, col)
            self.insert_water_diagonals(row, col)
            # place Middle piece & Water around it
            if self.board[row][col + 1] == "M":
                self.remaining_pieces["M"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col + 1] = "M"
            self.insert_water_ontop_below(row, col + 1)
            self.insert_water_diagonals(row, col)
            # place Middle piece & Water around it
            if self.board[row][col + 2] == "M":
                self.remaining_pieces["M"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col + 2] = "M"
            self.insert_water_ontop_below(row, col + 2)
            self.insert_water_diagonals(row, col)
            #place Right piece & Water around it
            if self.board[row][col + 3] == "R":
                self.remaining_pieces["TBRL"] += 1 # if piece is already placed, (it is a hint acho) increase count of edge pieces, to offset the one that will be removed later
            self.board[row][col + 3] = "R"
            self.insert_water_ontop_below(row, col + 3)
            self.insert_water_right(row, col + 3)
            self.insert_water_diagonals(row, col)
            # decrease count of edge & middle pieces
            self.remaining_ships["1x4"] -= 1
            self.remaining_pieces["TBRL"] -= 2
            self.remaining_pieces["M"] -= 2
        
        self.fill_completed_row_col() # Fill rows and columns that are completed with water


class Bimaru(Problem):
    def __init__(self, board, remaining_pieces, row_hints, col_hints, unfinished_hints, remaining_ships, empty_cells_values):
        """O construtor especifica o estado inicial."""
        # number of positions in the row / column with a ship cell
        self.row_hints = row_hints 
        self.col_hints = col_hints
        board_object = Board(board, remaining_pieces, unfinished_hints, remaining_ships, self) #Criar o Board inicial, passando o problema Bimaru para poder aceder às hints
        board_object.fill_water_around_hints() # Fill water around hints
        self.empty_cells_values = empty_cells_values
        self.state = BimaruState(board_object)
        super().__init__(self.state)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        
        actions = []
        # TODO: Debug
        # print("\n\n")
        # print("CHOSEN STATE ID:", state.id)
        # count = np.count_nonzero(state.board.board == "")
        # print("Empty cells heuristic:", count)
        # print(state.board.board)
        # print("\n\n")

        # for empty_cell in self.empty_cells_values:
        #     empty_cell_row = empty_cell[0]
        #     empty_cell_col = empty_cell[1]
        #     if(state.board.board[empty_cell_row][empty_cell_col] != ""):
        #         self.empty_cells_values.remove(empty_cell)
        # self.empty_cells_values.sort(key=lambda x: x[2])

        # First Fill all Hints
        if len(state.board.unfinished_hints) > 0:
            actions = state.board.hint_actions()
            return actions
        
        # After placing all hints, try to place ships on empty cells
            # Try to place a Ships (Horizontal and Vertical): 1x1, 1x2, 1x3, 1x4 (Centered on the topmost/left most piece)
                # Start by placing first the bigger pieces and only then place the smaller ones
        # for i in range(len(self.empty_cells_values)):
        #     row = self.empty_cells_values[i][0]
        #     col = self.empty_cells_values[i][1]
        #     if sum(state.board.remaining_ships.values()) >= np.count_nonzero(state.board.board == ""): # if the number of pieces to place is larger than the number of empty cells, then place pieces
                    #Ta a ver o numero de barcos e nao pieces   --  mas se o numero de barcos for maior que o numero de celulas vazias, entao nao faz sentrido continuar, deviia ser ao contrario?
                    # Isto assim parece que vai dar ao mesmo, poreque o resultado é também todas as possiveis colocações de barccos de 4, 3 etc. nas empty cells.
        #         if state.board.remaining_ships["1x4"] > 0:
        #             # try to plae a 1x4 ship vertivaly (topmost square on current cell)
        #             if state.board.check_place_1x4_vertical(row,col):
        #                 actions.append((row, col, "1x4_vertical", "empty", 1, 1)) # 1, 1 are just place holders, those slots are only ussed for hints to know where the original hint was
        #             # try to plae a 1x4 ship horizontaly (leftmost square on current cell)
        #             if state.board.check_place_1x4_horizontal(row,col):
        #                 actions.append((row, col, "1x4_horizontal", "empty", 1, 1))
        #         elif  state.board.remaining_ships["1x3"] > 0:
        #             # try to plae a 1x3 ship vertivaly (topmost square on current cell)
        #             if state.board.check_place_1x3_vertical(row,col):
        #                 actions.append((row, col, "1x3_vertical", "empty", 1, 1))
        #             # try to plae a 1x3 ship horizontaly (leftmost square on current cell)
        #             if state.board.check_place_1x3_horizontal(row,col):
        #                 actions.append((row, col, "1x3_horizontal", "empty", 1, 1))
        #         elif  state.board.remaining_ships["1x2"] > 0:
        #             # try to plae a 1x2 ship vertivaly (topmost square on current cell)
        #             if state.board.check_place_1x2_vertical(row,col):
        #                 actions.append((row, col, "1x2_vertical", "empty", 1, 1))
        #             # try to plae a 1x2 ship horizontaly (leftmost square on current cell)
        #             if state.board.check_place_1x2_horizontal(row,col):
        #                 actions.append((row, col, "1x2_horizontal", "empty", 1, 1))
        #         elif  state.board.remaining_ships["1x1"] > 0:
        #             # try to plae a 1x1 ship (on current cell)
        #             if state.board.check_place_1x1(row,col):
        #                 actions.append((row, col, "1x1", "empty", 1, 1))

        for row in range(10):
            for col in range(10):
                if state.board.get_value(row, col) == "":
                    if state.board.remaining_ships["1x4"] > 0:
                        # try to plae a 1x4 ship vertivaly (topmost square on current cell)
                        if state.board.check_place_1x4_vertical(row,col):
                            actions.append((row, col, "1x4_vertical", "empty", 1, 1)) # 1, 1 are just place holders, those slots are only ussed for hints to know where the original hint was
                        # try to plae a 1x4 ship horizontaly (leftmost square on current cell)
                        if state.board.check_place_1x4_horizontal(row,col):
                            actions.append((row, col, "1x4_horizontal", "empty", 1, 1))
                    elif  state.board.remaining_ships["1x3"] > 0:
                        # try to plae a 1x3 ship vertivaly (topmost square on current cell)
                        if state.board.check_place_1x3_vertical(row,col):
                            actions.append((row, col, "1x3_vertical", "empty", 1, 1))
                        # try to plae a 1x3 ship horizontaly (leftmost square on current cell)
                        if state.board.check_place_1x3_horizontal(row,col):
                            actions.append((row, col, "1x3_horizontal", "empty", 1, 1))
                    elif  state.board.remaining_ships["1x2"] > 0:
                        # try to plae a 1x2 ship vertivaly (topmost square on current cell)
                        if state.board.check_place_1x2_vertical(row,col):
                            actions.append((row, col, "1x2_vertical", "empty", 1, 1))
                        # try to plae a 1x2 ship horizontaly (leftmost square on current cell)
                        if state.board.check_place_1x2_horizontal(row,col):
                            actions.append((row, col, "1x2_horizontal", "empty", 1, 1))
                    elif  state.board.remaining_ships["1x1"] > 0:
                        # try to plae a 1x1 ship (on current cell)
                        if state.board.check_place_1x1(row,col):
                            actions.append((row, col, "1x1", "empty", 1, 1))
        return actions 
        
    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        row, col, ship, type, row_hint, col_hint = action
        # coordinates = self.determine_ship_coordinates(row, col, ship)
        # self.update_empty_cells_values(coordinates)
        new_board = copy.deepcopy(state.board)
        new_state = BimaruState(new_board)
        if type == "hint":
            new_state.board.unfinished_hints.remove((row_hint, col_hint))
        new_state.board.insert_ship(row, col, ship)
        
        # TODO: DEBUG
        #print("State ID:", new_state.id)
        #count = np.count_nonzero(new_state.board.board == "") + state.board.get_remaining_pieces()
        #print("Empty cells heuristic:", count)
        #print("Remaining Pieces:", new_state.board.get_remaining_pieces())
        #print("Remaining Ships:", new_state.board.remaining_ships)
        #print(new_state.board.board)
        #print(action)
        #print("\n\n")

        return new_state
    
    def determine_ship_coordinates(self, row, col, ship):
        coordinates = []
        if ship == "1x4_vertical":
            coordinates.append((row, col))
            coordinates.append((row + 1, col))
            coordinates.append((row + 2, col))
            coordinates.append((row + 3, col))
        elif ship == "1x4_horizontal":
            coordinates.append((row, col))
            coordinates.append((row, col + 1))
            coordinates.append((row, col + 2))
            coordinates.append((row, col + 3))
        elif ship == "1x3_vertical":
            coordinates.append((row, col))
            coordinates.append((row + 1, col))
            coordinates.append((row + 2, col))
        elif ship == "1x3_horizontal":
            coordinates.append((row, col))
            coordinates.append((row, col + 1))
            coordinates.append((row, col + 2))
        elif ship == "1x2_vertical":
            coordinates.append((row, col))
            coordinates.append((row + 1, col))
        elif ship == "1x2_horizontal":
            coordinates.append((row, col))
            coordinates.append((row, col + 1))
        elif ship == "1x1":
            coordinates.append((row, col))
        return coordinates


    def update_empty_cells_values(self, coordinates):
        for i in range(len(coordinates)):
            row, col = coordinates[i]
            for empty_cell in self.empty_cells_values:
                if empty_cell[0] == row and empty_cell[1] == col:
                    self.empty_cells_values.remove(empty_cell)
                    temp = list(empty_cell)
                    temp[2] = temp[2] + 1
                    self.empty_cells_values.append(tuple(temp))
                    break
        self.empty_cells_values.sort(key=lambda x: x[2])


    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        return state.board.get_remaining_pieces() == 0


    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return (np.count_nonzero(node.state.board.board == "") + node.state.board.get_remaining_pieces()) #+ node.state.board.get_smallest_row_col_remaining_pieces()


if __name__ == "__main__":
    # Ler o ficheiro do standard input, 
    board, remaining_pieces, row_hints, col_hints, unfinished_hints, remaining_ships = Board.parse_instance()
    first_board = copy.deepcopy(board)
    # Criar uma instância do problema Bimaru,
    empty_cells_values = []
    for i in range(10):
        for j in range(10):
            if(board[i][j] == ''):
                empty_cells_values.append((i, j, 0))
    problem = Bimaru(copy.deepcopy(board), remaining_pieces, row_hints, col_hints, unfinished_hints, remaining_ships, empty_cells_values)

    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    goal_node = astar_search(problem)
    # Imprimir para o standard output no formato indicado.
    if goal_node != None:
        solved_board = np.where(goal_node.state.board.board  == 'W', '.', goal_node.state.board.board)
        for i in range(10):
            for j in range(10):
                if solved_board[i][j] != '.' and first_board[i][j] == '':
                    solved_board[i][j] = solved_board[i][j].lower()
                elif first_board[i][j] != '':
                    solved_board[i][j] = first_board[i][j]
                print(solved_board[i][j], end='')
            print(end='\n')
    else:
        print("No solution found")

