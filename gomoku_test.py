import gomoku
import unittest


class TestGomoku(unittest.TestCase):

    def test_checkwinner(self):
        fake_board = [0, 0, 0, 0, 0,
                      1, 1, 1, 1, 0,
                      2, 2, 2, 2, 2,
                      0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0]
        g = gomoku.Gomoku(fake_board, 5, 5)
        self.assertEqual(g.check_winner(), 2)

        fake_board[5 * 2 + 1] = 0
        g = gomoku.Gomoku(fake_board, 5, 5)
        g.size_x = 5
        g.size_y = 5
        self.assertEqual(g.check_winner(), 0)

        for i in range(0, 5):
            fake_board[i * 5 + i] = 1
        g = gomoku.Gomoku(fake_board, 5, 5)
        g.size_x = 5
        g.size_y = 5
        self.assertEqual(g.check_winner(), 1)

    def test_countrow(self):
        fake_board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0,
                      0, 0, 0, 2, 2, 2, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        
        g = gomoku.Gomoku(fake_board, 12, 12)
        print(g.count_board(2))
        g.put_p(5,6,1)
        print(g.count_board(1))
        g.put_p(5,6,2)
        print(g.count_board(2))

        self.assertTrue(g.row_startswith(0b101101,0b1011))
        self.assertEqual(g.row_reverse(0b11011),0b11101)
        self.assertEqual(g.row_splice(0b1101001,3,6),0b1001)
        self.assertEqual(g.row_add_p(0b1001,1),0b10011)
        self.assertEqual(g.row_add_p(0b11,0),0b110)

        self.assertEqual(g.count_consec_row(0b1001110),g.patterns[0b101110])
        self.assertEqual(g.count_consec_row(0b10101110),g.patterns[0b101110]+g.patterns[0b101])


if __name__ == '__main__':
    unittest.main()
