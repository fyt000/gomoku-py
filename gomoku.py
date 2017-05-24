import math
import time
import numpy as np
from functools import reduce

global_row_eval = {}
global_sub_row_eval = {}


class Gomoku(object):

    def __init__(self, board_list, size_x=None, size_y=None, global_eval_dict=None, patterns=None):
        self.board = np.array(board_list)

        if patterns is None:
            self.patterns = {}
            self.add_p(0b1010, 1)
            self.add_p(0b101, 2) #0b101
            self.add_p(0b10110, 20) #0b10110
            self.add_p(0b101010, 20) #0b101010
            self.add_p(0b1011, 3) #0b1011
            self.add_p(0b101110, 1500) #0b101110
            self.add_p(0b10111, 20) #0b10111
            self.add_p(0b101011, 20) #0b101011
            self.add_p(0b1011110, 8000) #0b1011110
            self.add_p(0b10110110, 1500) #0b10110110
            self.add_p(0b10111010, 20) #0b10111010
            self.add_p(0b101111, 20) #0b101111
            self.add_p(0b1011011, 20) #0b1011011
            self.add_p(0b1011101, 20) #0b1011101
            self.add_p(0b111111, 100000) #0b111111
            self.add_p(0b10111110, 100000) #0b10111110
            self.add_p(0b1011111, 100000) #0b1011111

        self.max_pattern_len = reduce(
            (lambda x, y: max(x, y.bit_length()-1)), self.patterns.keys(), 0)
        self.size_x = size_x or 15
        self.size_y = size_y or 15

    def add_p(self, p, v):
        self.patterns[p] = v

    def __get_xy(self, index):
        return (math.floor(index / self.size_y), index % self.size_y)

    def __get_idx(self, x, y):
        return x * self.size_y + y

    def __get_p(self, x, y):
        return self.board[self.__get_idx(x, y)]

    def __put_p(self, x, y, p):
        self.board[self.__get_idx(x, y)] = p

    def __empty_p(self):
        return 1

    def row_add_p(self, row, p):
        return (row << 1) | p

    def check_winner(self):
        if len(self.board) != self.size_x * self.size_y:
            return 0
        # up, up-right, right, down-right, down, down-left, left, up-left
        dirx = [0, 1, 1, 1, 0, -1, -1, -1]
        diry = [-1, -1, 0, 1, 1, 1, 0, -1]
        for x in range(0, self.size_x):
            for y in range(0, self.size_y):
                cur_piece = self.__get_p(x, y)
                if cur_piece == 0:
                    continue
                for dx, dy in zip(dirx, diry):
                    count = 0
                    nx = x
                    ny = y
                    for j in range(0, 4):
                        nx = nx + dx
                        ny = ny + dy
                        if nx >= 0 and nx < 15 and ny >= 0 and ny < 15:
                            if self.__get_p(nx, ny) != cur_piece:
                                break
                            else:
                                count += 1
                        else:
                            break
                    if count == 4:
                        return cur_piece
        return 0

    def row_is_set(self, row, idx, rlen):
        return (row & (1 << (rlen - 1 - idx))) != 0

    def row_splice(self, row, pos, rlen):
        # basically zeroing out stuff and stick a 1
        row &= ~(1 << rlen)
        for i in range(0, pos):
            row &= ~(1 << rlen - 1 - i)
        row |= 1 << (rlen - pos)
        return row

    def row_length(self, row):
        return row.bit_length() - 1

    def row_startswith(self, row, p):
        len1 = row.bit_length()
        len2 = p.bit_length()
        if len1 < len2:
            return False
        return (row >> (len1 - len2)) == p

    def row_reverse(self, row):
        rlen = row.bit_length()
        backward_row = self.__empty_p()
        for i in range(0, rlen - 1):
            if (row & (1 << i)) != 0:
                backward_row = (backward_row << 1) | 1
            else:
                backward_row = (backward_row << 1)
        return backward_row

    # can probably preprocess this
    # 2^15 combinations, seems resonable
    def row_dper(self, row, dp, idx):
        # print(row)
        cur_max = 0
        row_len = self.row_length(row)
        if row_len < 3:
            dp[idx] = 0
            return 0
        saved_val = dp[idx]
        if not saved_val == -1:
            return saved_val

        # skip empty spaces
        if not self.row_is_set(row, 0, row_len) and not self.row_is_set(row, 1, row_len):
            return self.row_dper(self.row_splice(row,1,row_len), dp, idx + 1)

        for p, p_eval in self.patterns.items():
            p_len = self.row_length(p)
            rp = self.row_reverse(p)
            if self.row_startswith(row,rp):
                # print("detected rp",r_row,rp)
                cur_val = p_eval + \
                    self.row_dper(self.row_splice(row,p_len,row_len), dp, idx + p_len)
                cur_max = max(cur_val, cur_max)
            if self.row_startswith(row,p):
                cur_val = p_eval + \
                    self.row_dper(self.row_splice(row,p_len,row_len), dp, idx + p_len)
                cur_max = max(cur_val, cur_max)

        # move on, do nothing
        cur_val = self.row_dper(self.row_splice(row,1,row_len), dp, idx + 1)
        cur_max = max(cur_val, cur_max)

        dp[idx] = cur_max
        return cur_max

    def init_dp(self, row):
        # length of row
        rlen = row.bit_length()  # good, life just got easier
        dp = np.full(rlen + self.max_pattern_len, -1, dtype=int)
        for i in range(rlen, rlen + self.max_pattern_len):
            dp[i] = 0
        return (dp, rlen)

    # new matching algorithm
    # dynamic programming using heuristic to select the best for a row
    # aggregate for all rows of different directions
    def count_consec_row(self, row):

        if row <= (1 << 5):
            return 0

        if row in global_sub_row_eval:
            return global_sub_row_eval[row]

        # does not contain any 'O'
        for i in range(0, 17):
            if row == 1 << i:
                global_sub_row_eval[row] = 0
                return 0

        (dp_forward, rlen) = self.init_dp(row)
        dp_backward = dp_forward[:]

        forward = self.row_dper(row, dp_forward, 0)

        backward_row = self.__empty_p()
        for i in range(0, rlen):
            if row & (1 << i) == 1:
                backward_row = (backward_row << 1) & 1
            else:
                backward_row = (backward_row << 1)
        backward = self.row_dper(backward_row, dp_backward, 0)

        ret = max(forward, backward)
        global_sub_row_eval[row] = ret
        return ret

    def count_rowx(self, x, y, dx, dy, cur):
        opponent = 2 if cur == 1 else 1
        row = self.__empty_p()
        cur_score = 0
        for i in range(0, max(self.size_x, self.size_y)):
            if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y:
                return cur_score
            p = self.__get_p(x, y)
            if p == opponent:
                cur_score += self.count_consec_row(row)
                row = self.__empty_p()
                continue
            row = self.row_add_p(row, 0 if p == 0 else 1)
            x += dx
            y += dy
        return cur_score

    def count_board(self, cur):
        #dirx = [0, 1, 1, 1]
        #diry = [1, -1, 1, 0]
        val = 0
        # vertical
        for i in range(0, self.size_y):
            val += self.count_rowx(0, i, 1, 0, cur)

        # horizontal
        for i in range(0, self.size_y):
            val += self.count_rowx(i, 0, 0, 1, cur)

        #diagonal \
        for i in range(0, self.size_y):
            val += self.count_rowx(0, i, 1, 1, cur)
        for i in range(1, self.size_x):
            val += self.count_rowx(i, 0, 1, 1, cur)

        # diagonal /
        for i in range(0, self.size_y):
            val += self.count_rowx(0, i, 1, -1, cur)
        for i in range(1, self.size_x):
            val += self.count_rowx(i, self.size_y - 1, 1, -1, cur)

        return val

    def get_next_move(self, cur):
        opponent = 2 if cur == 1 else 1
        (v, x, y) = self.mini_max(2, True, cur, opponent)
        return (x, y)

    def get_best_moves(self, cur, opponent):
        l = []
        for x in range(0, self.size_x):
            for y in range(0, self.size_y):
                if self.__get_p(x, y) == 0:
                    t = time.process_time()
                    self.__put_p(x, y, cur)
                    score = self.count_board(cur)
                    self.__put_p(x, y, opponent)
                    score += self.count_board(opponent)
                    l.append((score, x, y))
                    self.__put_p(x, y, 0)
                    print("took ",time.process_time() - t)

        l = sorted(l, key=lambda x: x[0], reverse=True)
        print(l[:10])
        return l[:10]

    def mini_max(self, depth, maximizing, cur, opponent):
        if depth == 0 or not self.check_winner() == 0:
            return (self.count_board(cur) - self.count_board(opponent), -1, -1)

        best_x = -1
        best_y = -1
        best_val = 0

        if maximizing:
            best_val = -999999
            # limit to the center
            for (s, x, y) in self.get_best_moves(cur, opponent):
                self.__put_p(x, y, cur)
                (v, b_x, b_y) = self.mini_max(depth - 1, False, cur, opponent)
                if v > best_val:
                    best_x = x
                    best_y = y
                    best_val = v
                self.__put_p(x, y, 0)
        else:
            best_val = 999999
            for (s, x, y) in self.get_best_moves(opponent, cur):
                self.__put_p(x, y, opponent)
                (v, b_x, b_y) = self.mini_max(depth - 1, True, cur, opponent)
                if v < best_val:
                    best_x = x
                    best_y = y
                    best_val = v
                self.__put_p(x, y, 0)

        return (best_val, best_x, best_y)
