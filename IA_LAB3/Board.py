import random

from ChessPiece import *
from copy import deepcopy


class Board:
    whites = []
    blacks = []

    def __init__(self, game_mode, ai=False, depth=2, log=False):  # game_mode == 0: whites down/blacks up
        self.zobrist_keys = {}
        for piece_type in ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']:
            for color in ['white', 'black']:
                for row in range(8):
                    for col in range(8):
                        key = random.getrandbits(64)
                        self.zobrist_keys[(piece_type, color, row, col)] = key

        self.board = []
        self.game_mode = game_mode
        self.depth = depth
        self.ai = ai
        self.log = log

    def initialize_board(self):
        for i in range(8):
            self.board.append(['empty-block' for _ in range(8)])

    def place_pieces(self):
        self.board.clear()
        self.whites.clear()
        self.blacks.clear()
        self.initialize_board()
        self.whiteKing = King('white', 0, 4, '\u265A')
        self.blackKing = King('black', 7, 4, '\u2654')
        for j in range(8):
            self[1][j] = Pawn('white', 1, j, '\u265F')
            self[6][j] = Pawn('black', 6, j, '\u2659')
        self[0][0] = Rook('white', 0, 0, '\u265C')
        self[0][7] = Rook('white', 0, 7, '\u265C')
        self[0][1] = Knight('white', 0, 1, '\u265E')
        self[0][6] = Knight('white', 0, 6, '\u265E')
        self[0][2] = Bishop('white', 0, 2, '\u265D')
        self[0][5] = Bishop('white', 0, 5, '\u265D')
        self[0][3] = Queen('white', 0, 3, '\u265B')
        self[0][4] = self.whiteKing
        self[7][0] = Rook('black', 7, 0, '\u2656')
        self[7][7] = Rook('black', 7, 7, '\u2656')
        self[7][1] = Knight('black', 7, 1, '\u2658')
        self[7][6] = Knight('black', 7, 6, '\u2658')
        self[7][2] = Bishop('black', 7, 2, '\u2657')
        self[7][5] = Bishop('black', 7, 5, '\u2657')
        self[7][3] = Queen('black', 7, 3, '\u2655')
        self[7][4] = self.blackKing

        self.save_pieces()

        if self.game_mode != 0:
            self.reverse()

    def get_hash_key(self):
        hash_key = 0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if isinstance(piece, ChessPiece):
                    piece_type = type(piece).__name__
                    color = piece.color
                    hash_key ^= self.zobrist_keys[(piece_type, color, row, col)]
        return hash_key

    def save_pieces(self):
        for i in range(8):
            for j in range(8):
                if isinstance(self[i][j], ChessPiece):
                    if self[i][j].color == 'white':
                        self.whites.append(self[i][j])
                    else:
                        self.blacks.append(self[i][j])

    def make_move(self, piece, x, y, keep_history=False):  # history is logged when AI searches for moves
        old_x = piece.x
        old_y = piece.y
        if keep_history:
            self.board[old_x][old_y].set_last_eaten(self.board[x][y])
        else:
            if isinstance(self.board[x][y], ChessPiece):
                if self.board[x][y].color == 'white':
                    self.whites.remove(self.board[x][y])
                else:
                    self.blacks.remove(self.board[x][y])
        self.board[x][y] = self.board[old_x][old_y]
        self.board[old_x][old_y] = 'empty-block'
        self.board[x][y].set_position(x, y, keep_history)

    def unmake_move(self, piece):
        x = piece.x
        y = piece.y
        self.board[x][y].set_old_position()
        old_x = piece.x
        old_y = piece.y
        self.board[old_x][old_y] = self.board[x][y]
        self.board[x][y] = piece.get_last_eaten()

    def reverse(self):
        self.board = self.board[::-1]
        for i in range(8):
            for j in range(8):
                if isinstance(self.board[i][j], ChessPiece):
                    piece = self.board[i][j]
                    piece.x = i
                    piece.y = j

    def __getitem__(self, item):
        return self.board[item]

    def has_opponent(self, piece, x, y):
        if not self.is_valid_move(x, y):
            return False
        if isinstance(self.board[x][y], ChessPiece):
            return piece.color != self[x][y].color
        return False

    def has_friend(self, piece, x, y):
        if not self.is_valid_move(x, y):
            return False
        if isinstance(self[x][y], ChessPiece):
            return piece.color == self[x][y].color
        return False

    @staticmethod
    def is_valid_move(x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def has_empty_block(self, x, y):
        if not self.is_valid_move(x, y):
            return False
        return not isinstance(self[x][y], ChessPiece)

    def get_player_color(self):
        if self.game_mode == 0:
            return 'white'
        return 'black'

    def king_is_threatened(self, color, move=None):
        if color == 'white':
            enemies = self.blacks
            king = self.whiteKing
        else:
            enemies = self.whites
            king = self.blackKing
        threats = []
        for enemy in enemies:
            moves = enemy.get_moves(self)
            if (king.x, king.y) in moves:
                threats.append(enemy)
        if move and len(threats) == 1 and threats[0].x == move[0] and threats[0].y == move[1]:
            return False
        return True if len(threats) > 0 else False

    def is_terminal(self):
        terminal1 = self.white_won()
        terminal2 = self.black_won()
        terminal3 = self.draw()
        return terminal1 or terminal2 or terminal3

    def draw(self):
        if not self.king_is_threatened('white') and not self.has_moves('white'):
            return True
        if not self.king_is_threatened('black') and not self.has_moves('black'):
            return True
        if self.insufficient_material():
            return True
        return False

    def white_won(self):
        if self.king_is_threatened('black') and not self.has_moves('black'):
            return True
        return False

    def black_won(self):
        if self.king_is_threatened('white') and not self.has_moves('white'):
            return True
        return False

    def has_moves(self, color):
        total_moves = 0
        for i in range(8):
            for j in range(8):
                if isinstance(self[i][j], ChessPiece) and self[i][j].color == color:
                    piece = self[i][j]
                    total_moves += len(piece.filter_moves(piece.get_moves(self), self))
                    if total_moves > 0:
                        return True
        return False

    def insufficient_material(self):
        total_white_knights = 0
        total_black_knights = 0
        total_white_bishops = 0
        total_black_bishops = 0
        total_other_white_pieces = 0
        total_other_black_pieces = 0

        for piece in self.whites:
            if piece.type == 'Knight':
                total_white_knights += 1
            elif piece.type == 'Bishop':
                total_white_bishops += 1
            elif piece.type != 'King':
                total_other_white_pieces += 1

        for piece in self.blacks:
            if piece.type == 'Knight':
                total_black_knights += 1
            elif piece.type == 'Bishop':
                total_black_bishops += 1
            elif piece.type != 'King':
                total_other_black_pieces += 1

        weak_white_pieces = total_white_bishops + total_white_knights
        weak_black_pieces = total_black_bishops + total_black_knights

        if self.whiteKing and self.blackKing:
            if weak_white_pieces + total_other_white_pieces + weak_black_pieces + total_other_black_pieces == 0:
                return True
            if weak_white_pieces + total_other_white_pieces == 0:
                if weak_black_pieces == 1:
                    return True
            if weak_black_pieces + total_other_black_pieces == 0:
                if weak_white_pieces == 1:
                    return True
            if len(self.whites) == 1 and len(self.blacks) == 16 or len(self.blacks) == 1 and len(self.whites) == 16:
                return True
            if total_white_knights == weak_white_pieces + total_other_white_pieces and len(self.blacks) == 1:
                return True
            if total_black_knights == weak_black_pieces + total_other_black_pieces and len(self.whites) == 1:
                return True
            if (weak_white_pieces == weak_black_pieces == 1
                    and total_other_white_pieces == total_other_black_pieces == 0):
                return True

    def evaluate(self):
        white_points = 0
        black_points = 0
        for i in range(8):
            for j in range(8):
                if isinstance(self[i][j], ChessPiece):
                    piece = self[i][j]
                    if piece.color == 'white':
                        white_points += piece.get_score()
                        white_points += piece.x + piece.y
                        if piece.type == 'King':
                            white_points += self.calculate_king_safety(piece)
                        white_points += len(piece.filter_moves(piece.get_moves(self), self))*0.5
                    else:
                        black_points += piece.get_score()
                        black_points += piece.x + piece.y
                        if piece.type == 'King':
                            black_points += self.calculate_king_safety(piece)
                        black_points += len(piece.filter_moves(piece.get_moves(self), self))*0.5

        if self.game_mode == 0:
            return black_points - white_points
        return white_points - black_points

    def calculate_king_safety(self, king):
        safety_value = 0
        king_x, king_y = king.x, king.y
        player_color = king.color
        # opponent_color = 'white' if player_color == 'black' else 'black'

        # Evaluate pawn shield in front of the king
        pawn_shield = self.get_pawn_shield(king_x, king_y, player_color)
        safety_value += len(pawn_shield)  # Add value based on the size of the pawn shield

        # Evaluate open files near the king
        open_files = self.get_open_files(king_x, king_y, player_color)
        safety_value += len(open_files) * 0.5  # Add value for each open file near the king

        # Evaluate piece placement near the king (e.g., knights, bishops)
        piece_placement_value = self.evaluate_piece_placement(king_x, king_y, player_color)
        safety_value += piece_placement_value

        return safety_value

    def get_pawn_shield(self, king_x, king_y, color):
        shield = []
        king_row, king_col = king_x, king_y
        pawn_offsets = [(1, -1), (1, 0), (1, 1)] if color == 'White' else [(-1, -1), (-1, 0), (-1, 1)]
        for offset in pawn_offsets:
            row, col = king_row + offset[0], king_col + offset[1]
            if self.is_valid_move(row, col) and isinstance(self[row][col], Pawn) and self[row][col].color == color:
                shield.append((row, col))
        return shield

    def get_open_files(self, king_x, king_y, color):
        files = set()
        king_row, king_col = king_x, king_y
        for col in range(8):
            if col != king_col:
                open_file = True
                for row in range(8):
                    if self[row][col] is not None:
                        open_file = False
                        break
                if open_file:
                    files.add(col)
        return files

    def evaluate_piece_placement(self, king_x, king_y, color):
        placement_value = 0
        king_row, king_col = king_x, king_y
        for i in range(-1, 2):
            for j in range(-1, 2):
                row, col = king_row + i, king_col + j
                if self.is_valid_move(row, col) and isinstance(self[row][col], (Knight, Bishop)):
                    placement_value += 0.5 if self[row][col].color == color else -0.5
        return placement_value

    # def get_opponent_threats(self, king_x, king_y, opponent_color):
    #     threats = []
    #     for i in range(8):
    #         for j in range(8):
    #             piece = self[i][j]
    #             if isinstance(piece, ChessPiece) and piece.color == opponent_color:
    #                 moves = piece.get_moves(self)
    #                 for move in moves:
    #                     if move[1] == king_x, king_y:
    #                         threats.append(piece)
    #     return threats

    def calculate_pawn_structure_value(self):
        pawn_structure_value = 0
        for i in range(8):
            for j in range(8):
                piece = self[i][j]
                if isinstance(piece, Pawn) and piece.color == self.get_player_color():
                    # penalize isolated pawns
                    if self.is_isolated_pawn(piece, i, j):
                        pawn_structure_value -= 0.5
                    # penalize doubled pawns
                    if self.is_doubled_pawn(piece, i, j):
                        pawn_structure_value -= 0.5
        return pawn_structure_value

    # helper methods for pawn structure evaluation
    def is_isolated_pawn(self, pawn, row, col):
        # Check if the pawn is isolated (no friendly pawns on adjacent files)
        return self[row - 1][col - 1] is None and self[row - 1][col + 1] is None

    def is_doubled_pawn(self, pawn, row, col):
        # check if the pawn is doubled (another pawn of the same color on the same file)
        for i in range(8):
            if isinstance(self[i][col], Pawn) and self[i][col].color == pawn.color and i != row:
                return True
        return False

    def __str__(self):
        return str(self[::-1]).replace('], ', ']\n')

    def __repr__(self):
        return 'Board'

    def unicode_array_repr(self):
        data = deepcopy(self.board)
        for idx, row in enumerate(self.board):
            for i, p in enumerate(row):
                if isinstance(p, ChessPiece):
                    un = p.unicode
                else:
                    un = '\u25AF'
                data[idx][i] = un
        return data[::-1]

    def get_king(self, piece):
        if piece.color == 'white':
            return self.whiteKing
        return self.blackKing
