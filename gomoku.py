import math
import time
import numpy as np
from functools import reduce


global_eval_dict={}

class Gomoku(object):

    def __init__(self, board_list, size_x=None, size_y=None, global_eval_dict=None, patterns=None):
        self.board = np.array(board_list)

        if patterns is None:
            self.patterns={}
            self.add_p('_O_',3)
            self.add_p('_OX',1)
            self.add_p('_OO_',20)
            self.add_p('_O_O_',20)
            self.add_p('_OOX',3)
            self.add_p('_OOO_',1500)
            self.add_p('_OOOX',20)
            self.add_p('_O_OOX',20)
            self.add_p('_OOOO_',3000)
            self.add_p('_OO_OO_',2500)
            self.add_p('_OOO_O_',20)
            self.add_p('_OOOOX',20)
            self.add_p('_OO_OOX',20)
            self.add_p('_OOO_OX',20)
            self.add_p('XOOOOOX',100000)
            self.add_p('_OOOOO_',100000)
            self.add_p('_OOOOOX',100000)

        self.max_pattern_len = reduce(
            (lambda x, y: max(x, len(y))), self.patterns.keys(), 0)
        self.size_x = size_x or 15
        self.size_y = size_y or 15

    def add_p(self,p,v):
        self.patterns[p]=v

    def __get_xy(self, index):
        return (math.floor(index / self.size_y), index % self.size_y)

    def __get_idx(self, x, y):
        return x * self.size_y + y

    def __get_p(self, x, y):
        return self.board[self.__get_idx(x, y)]

    def __put_p(self,x,y,p):
        self.board[self.__get_idx(x,y)]=p

    # relative to cur, eg. if cur = 1 then pchar of 1 -> O,  pchar of 2 -> X
    def __get_pchar(self, x, y, cur):
        if x >= self.size_x or y >= self.size_y or x < 0 or y < 0:
            return 'X'  # hitting the boundary is equivalent to X
        v = self.__get_p(x, y)
        if v == 0:
            return '_'
        if v == cur:
            return 'O'
        else:
            return 'X'

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


    #can probably preprocess this
    #2^15 combinations, seems resonable
    def row_dper(self, row, dp, idx):
        # print(row)
        cur_max = 0
        if len(row) < 3:
            dp[idx] = 0
            return 0
        saved_val = dp[idx]
        if not saved_val == -1:
            return saved_val
        # skip empty spaces
        if row[0] == '_' and row[1] == '_':
            cur_max = self.row_dper(row[1:], dp, idx + 1)
        for p, p_eval in self.patterns.items():
            rp = p[::-1]
            r_len = len(p)
            r_row = row
            #the 2nd condition is to check that this is not the first call
            if rp[0] == 'X' and not row[0] == 'X':
                r_row = 'X' + r_row
                r_len -= 1
            if r_row.startswith(rp):
                # print("detected rp",r_row,rp)
                cur_val = p_eval + \
                    self.row_dper(row[r_len:], dp, idx + r_len)
                cur_max = max(cur_val, cur_max)
            if row.startswith(p):
                cur_val = p_eval + \
                    self.row_dper(row[len(p):], dp, idx + len(p))
                cur_max = max(cur_val, cur_max)
        if row.startswith('X'):
            cur_val = self.row_dper(row[1:],dp,idx+1)
            cur_max = max(cur_val, cur_max)

        dp[idx] = cur_max
        return cur_max

    # # this does not work. but anyway
    # def row_backtrack(self, row, dp, idx, counts):
    #     if len(row) < 3:
    #         return
    #     if row[0] == '_' and row[1] == '_':
    #         self.row_backtrack(row[1:], dp, idx + 1, counts)
    #         return
    #     cur_val = dp[idx]

    #     if row.startswith('X'):
    #         if cur_val == dp[idx+1]:
    #              self.row_backtrack(row[1:],dp,idx+1,counts)
    #              return

    #     #fix to check that pattern matches first
    #     for p, p_eval in self.patterns.items():
    #         if (row.startswith(p) or row.startswith(p[::-1])) and cur_val == dp[idx + len(p)] + p_eval:
    #             counts[p] += 1
    #             self.row_backtrack(row[len(p):],dp,idx+len(p),counts)
    #             break
    #         if p[::-1].startswith('X') and not row[0] == 'X':
    #             if ('X'+row).startswith(p) and cur_val == dp[idx + len(p) - 1] + p_eval:
    #                 counts[p] += 1
    #                 self.row_backtrack(row[len(p)-1:], dp, idx+len(p)-1,counts)
    #                 break
        

    def init_dp(self,row):
        dp = []
        for c in row:
            dp.append(-1)                   
        # easy boundary checking
        for i in range(0, self.max_pattern_len):
            dp.append(0)
        
        return dp


    # new matching algorithm
    # dynamic programming using heuristic to select the best for a row
    # aggregate for all rows of different directions
    def count_consec_row(self, row):
        if row in global_eval_dict:
            return global_eval_dict[row]

        dp_forward = self.init_dp(row)
        dp_backward = dp_forward[:]
        row = 'X' + row + 'X'
        #print(row)
        forward = self.row_dper(row,dp_forward,0)
        backward = self.row_dper(row[::-1],dp_backward,0)
        ret = max(forward,backward)
        global_eval_dict[row] = ret
        return ret


    def count_row(self, row):
        pieces_list = row.split('X')
        val = 0
        for pieces in pieces_list:
            if len(pieces) < 5:
                continue
            val += self.count_consec_row(pieces)
        return val


    def grab_row(self,x,y,dx,dy,cur):
        row = self.__get_pchar(x,y,cur)
        for i in range(0,max(self.size_x,self.size_y)):
            x = x + dx
            y = y + dy
            if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y:
                return row
            row += self.__get_pchar(x,y,cur)
        return row

    def count_board(self,cur):
        #dirx = [0, 1, 1, 1]
        #diry = [1, -1, 1, 0]
        val = 0
        #vertical
        for i in range(0,self.size_y):
            val += self.count_row(self.grab_row(0,i,1,0,cur))
        
        #horizontal
        for i in range(0,self.size_y):
            val += self.count_row(self.grab_row(i,0,0,1,cur))

        #diagonal \
        for i in range(0, self.size_y):
            val += self.count_row(self.grab_row(0,i,1,1,cur))
        for i in range(1, self.size_x):
            val += self.count_row(self.grab_row(i,0,1,1,cur))

        #diagonal /
        for i in range(0, self.size_y):
            val += self.count_row(self.grab_row(0,i,1,-1,cur))
        for i in range(1, self.size_x):
            val += self.count_row(self.grab_row(i,self.size_y-1,1,-1,cur))

        return val

    def get_next_move(self,cur):
        print("global eval size ",len(global_eval_dict))
        opponent = 2 if cur == 1 else 1
        max_score = -999999
        cur_x=-1
        cur_y=-1
        for x in range(0,self.size_x):
            for y in range(0,self.size_y):
                if self.__get_p(x,y) == 0:
                    self.__put_p(x,y,cur)
                    score = self.count_board(cur)
                    if score > self.patterns['_OOOOO_']:
                        return (x, y)
                    self.__put_p(x,y,opponent)
                    score += self.count_board(opponent)*1.1
                    if score > max_score:
                        max_score = score
                        cur_x = x
                        cur_y = y
                    self.__put_p(x,y,0)
        return (cur_x,cur_y)


    # def score_board(self):
    #     score = 0
    #     for v in [1, 2]:
    #         score += self.evaluator(self.count_board(v)) * \
    #             (-1 if v == 1 else 1)
    #     return score
