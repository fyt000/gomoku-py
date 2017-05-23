import gomoku
import unittest


class TestGomoku(unittest.TestCase):

    # def test_checkwinner(self):
    #     fake_board = [0, 0, 0, 0, 0,
    #                   1, 1, 1, 1, 0,
    #                   2, 2, 2, 2, 2,
    #                   0, 0, 0, 0, 0,
    #                   0, 0, 0, 0, 0]
    #     g = gomoku.Gomoku(fake_board, 5, 5)
    #     self.assertEqual(g.check_winner(), 2)

    #     fake_board[5 * 2 + 1] = 0
    #     g = gomoku.Gomoku(fake_board, 5, 5)
    #     g.size_x = 5
    #     g.size_y = 5
    #     self.assertEqual(g.check_winner(), 0)

    #     for i in range(0, 5):
    #         fake_board[i * 5 + i] = 1
    #     g = gomoku.Gomoku(fake_board, 5, 5)
    #     g.size_x = 5
    #     g.size_y = 5
    #     self.assertEqual(g.check_winner(), 1)

    def test_countrow(self):
        fake_board = [0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0,
                      1, 1, 0, 1, 0, 0,
                      0, 0, 0, 1, 0, 0,
                      0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0]
        g = gomoku.Gomoku(fake_board, 6, 6)

        # counts = g.init_counts()
        # g.count_consec_row("__O___",counts)
        # counts = g.init_counts()
        # g.count_consec_row("_OOO__",counts)
        counts = g.init_counts()
        #val = g.count_consec_row('_O_O_OOO_',counts)
        val = g.count_consec_row('O_O_',counts)
        val = g.count_consec_row('_O_O_OOO_',counts)
        val = g.count_consec_row('_O_OOO_',counts)

        print(val)

        # counts = g.count_board(1)
        # self.assertEqual(counts['_OOX'], 1)

    # def test_countboard(self):
    #     fake_board = [0, 0, 0, 0, 0,
    #                   0, 0, 0, 1, 0,
    #                   1, 1, 0, 1, 0,
    #                   0, 0, 0, 1, 0,
    #                   0, 0, 0, 0, 0]
    #     g = gomoku.Gomoku(fake_board, 5, 5, ['_OOX', '_OOO_'])
    #     countMap = g.count_board(1, 0)
    #     self.assertEqual(countMap['_OOX'], 1)

    #     fake_board = [0, 0, 0, 0, 0,
    #                   0, 0, 0, 1, 0,
    #                   1, 1, 0, 1, 0,
    #                   0, 1, 0, 1, 0,
    #                   0, 0, 0, 0, 0]
    #     g = gomoku.Gomoku(fake_board, 5, 5, ['_OOX', '_OO_'])
    #     countMap = g.count_board(1, 0)
    #     self.assertEqual(countMap['_OOX'], 2)
    #     self.assertEqual(countMap['_OO_'],1)

    #     fake_board = [0, 0, 0, 0, 0,
    #                   0, 0, 0, 1, 0,
    #                   1, 1, 0, 1, 0,
    #                   0, 1, 0, 1, 0,
    #                   0, 0, 0, 0, 0]
    #     g = gomoku.Gomoku(fake_board, 5, 5)
    #     countMap = g.count_board(1, 0)
    #     self.assertEqual(countMap['_OOX'], 2)
    #     self.assertEqual(countMap['_OO_'],1)
    #     self.assertEqual(countMap['_OOO_'],1)
    #     self.assertEqual(countMap['_O_O_'],2)
    #     self.assertEqual(countMap['_O_OOX'], 1)
    #     self.assertEqual(countMap['_O_'], 10)
    #     self.assertEqual(countMap['_OX'], 1)
    #     # print(countMap)

    #     fake_board = [1, 0, 0, 0, 0,
    #                   1, 0, 0, 1, 0,
    #                   1, 1, 0, 1, 0,
    #                   1, 1, 0, 1, 0,
    #                   1, 2, 2, 2, 2]
    #     g = gomoku.Gomoku(fake_board, 5, 5)
    #     countMap = g.count_board(1)
    #     self.assertEqual(countMap['XOOOOOX'], 1)
    #     self.assertEqual(countMap['_OOOOOX'], 0)
    #     self.assertEqual(countMap['_OOOOO_'], 0)
    #     self.assertEqual(countMap['_O_OOX'], 3)
    #     countMap = g.count_board(2, 5)
    #     self.assertEqual(countMap['_OOOOX'],0)
    #     self.assertEqual(countMap['_OX'], 2)


if __name__ == '__main__':
    unittest.main()
