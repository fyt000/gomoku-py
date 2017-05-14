import math
import time


def get_xy(index):
    return (math.floor(index / 15), index % 15)


def get_idx(x, y):
    return x * 15 + y


def check_winner(board):
    # time.sleep(3)
    if len(board) != 15*15:
        return 0
    # up, up-right, right, down-right, down, down-left, left, up-left
    dirx = [0, 1, 1, 1, 0, -1, -1, -1]
    diry = [-1, -1, 0, 1, 1, 1, 0, -1]
    for i in range(0, 15*15):
        (x, y) = get_xy(i)
        if board[i] == 0:
            continue
        for d in range(0, len(dirx)):
            count = 0
            nx = x
            ny = y
            for j in range(0, 4):
                nx = nx + dirx[d]
                ny = ny + diry[d]
                if nx >= 0 and nx < 15 and ny >= 0 and ny < 15:
                    if board[get_idx(nx, ny)] != board[i]:
                        break
                    else:
                        count += 1
                else:
                    break
            if count == 4:
                return board[i]
    return 0
