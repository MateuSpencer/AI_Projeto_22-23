# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 100032 Mateus Spencer
# 00000 Nome2

import sys
import numpy as np
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

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Bimaru."""
    
    def __init__(self, board):
        self.board = board

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row][col]

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo, respectivamente."""
        above = self.get_value(row-1, col) if row > 0 else None
        below = self.get_value(row+1, col) if row < 9 else None
        return above, below

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita, respectivamente."""
        left = self.get_value(row, col-1) if col > 0 else None
        right = self.get_value(row, col+1) if col < 9 else None
        return left, right

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        rows_hints = []
        columns_hints = []
        board = np.zeros((10, 10), dtype=str) # empty cells represented with "0"

        for line in sys.stdin:
            # Split the line into parts by tabs
            parts = line.strip().split('\t')
            # Store the corresponding values
            if parts[0] == 'ROW':
                rows_hints = [int(x) for x in parts[1:]]
            elif parts[0] == 'COLUMN':
                columns_hints = [int(x) for x in parts[1:]]
            elif parts[0] == 'HINT':
                row, col, letter = int(parts[1]), int(parts[2]), parts[3]
                board[row][col] = letter
            elif parts[0].isdigit():
                pass

        return Board(board), rows_hints, columns_hints
        
    # TODO: também pode só ser updated quando um novo estado é criado
    def get_remaining_pieces(self): 
        """Retorna o número de peças que ainda faltam colocar no tabuleiro."""
        remaining_pieces = {"C": 4, "T": 2, "M": 3, "B": 2, "L": 3, "R": 3}
        for row in range(10):
            for col in range(10):
                piece = self.board.get_value(row, col)
                if piece in remaining_pieces:
                    remaining_pieces[piece] -= 1
        return sum(remaining_pieces.values())
    
    
    def place_C (self, row: int, col: int):
        #TODO: verificações à volta e na linha / coluna
        pass

    def place_M (self, row: int, col: int):
        #TODO: verificações à volta e na linha / coluna
        pass

    def place_T (self, row: int, col: int):
        #TODO: verificações à volta e na linha / coluna
        pass

    def place_B (self, row: int, col: int):
        #TODO: verificações à volta e na linha / coluna
        pass

    def place_R (self, row: int, col: int):
        #TODO: verificações à volta e na linha / coluna
        pass

    def place_L (self, row: int, col: int):
        #TODO: verificações à volta e na linha / coluna
        pass

def insert_piece(self, row: int, col: int, piece: str):
    """Insere uma peça na posição indicada."""
    self.board[row][col] = piece
    # possibilidade de decrementar peças por meter aqui

class Bimaru(Problem):
    def __init__(self, board: Board, rows_hints, columns_hints ):
        """O construtor especifica o estado inicial."""
        state = BimaruState(board)
        # number of positions in the row/column with a ship cell
        self.rows_hints = rows_hints 
        self.columns_hints = columns_hints

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        board = state.board
        for row in range(10):
            for col in range(10):
                if state.board.get_remaining_pieces() == 0:
                    break
                # if cell is empty
                if state.board.get_value(row, col) == "0":
                    #Two possible approaches:
                        # 1. Try to Place an Indicidual Piece: C, M, T, B, R, L
                        # 2. Try to place a Ship (Horizontal and Vertical): 1x1, 1x2, 1x3, 1x4 (Centered on the topmost/left most piece)
                    # Option 1.
                    #TODO: O board que estou a guardar pode ser assim ou tem de ser uma cópia?
                    # Try to place a C Piece
                    if state.board.place_C(row,col):
                        new_board = np.copy(board)
                        actions.append((new_board, row, col, "C"))
                    # Try to place a M Piece
                    if state.board.place_M(board, row,col):
                        new_board = np.copy(board)
                        actions.append((new_board, row, col, "M"))
                    # Try to place a T Piece
                    if state.board.place_T(row,col):
                        new_board = np.copy(board)
                        actions.append((new_board, row, col, "T"))
                    # Try to place a B Piece
                    if state.board.place_B(row,col):
                        new_board = np.copy(board)
                        actions.append((new_board, row, col, "B"))
                    # Try to place a L Piece
                    if state.board.place_L(row,col):
                        new_board = np.copy(board)
                        actions.append((new_board, row, col, "L"))
                    # Try to place a R Piece
                    if state.board.place_R(row,col):
                        new_board = np.copy(board)
                        actions.append((new_board, row, col, "R"))
                    #return a list of actions to append to the 
                    
        return actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        board, row, col, piece = action
        new_state = BimaruState(board)
        new_state.board.insert_piece(row, col, piece) # Update the piece on the Board
        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        return state.board.get_remaining_pieces() == 0

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return node.state.board.get_remaining_pieces()


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    board, rows_hints, columns_hints = Board.parse_instance()
    problem = Bimaru(board, rows_hints, columns_hints)
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
#goal_node = recursive_best_first_search(bimaru) #ou outra
    # Imprimir para o standard output no formato indicado.
#print(goal_node.state.board)
    
    pass
