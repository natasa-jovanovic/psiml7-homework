import numpy as np
from PIL import Image


def detect_piece(tile, figs):
    background = tile[0, 0, :]
    b, w, bg = (tile < background).all(axis=2).sum(), (tile > background).all(axis=2).sum(), (tile == background).all(
        axis=2).sum()

    min_diff = np.sqrt(b ** 2 + w ** 2)
    piece = 'b'
    bg_min = bg

    for fig in figs:
        br, wr, bgr = figs[fig][0][0], figs[fig][0][1], figs[fig][0][2]
        bg_diff = np.abs(bg - bgr)
        if bg_diff < bg_min: bg_min = bg_diff

    for fig in figs:
        br, wr, bgr = figs[fig][0][0], figs[fig][0][1], figs[fig][0][2]

        bg_diff = np.abs(bg - bgr)
        if bg_diff == bg_min:
            diff = np.sqrt((b - br) ** 2 + (w - wr) ** 2)
            if diff < min_diff:
                min_diff = diff
                piece = fig
    return piece


def move1(table, x, y):
    table[x, :] += 1
    table[:, y] += 1
    table[x, y] -= 2
    return table


def move2(table, x, y):
    for i in range(8):
        if x + i < 8 and y + i < 8: table[x + i, y + i] += 1
        if x - i >= 0 and y + i < 8: table[x - i, y + i] += 1
        if x - i >= 0 and y - i >= 0: table[x - i, y - i] += 1
        if x + i < 8 and y - i >= 0: table[x + i, y - i] += 1

    table[x, y] -= 4
    return table


def move3(table, x, y):
    if x - 2 >= 0 and y - 1 >= 0: table[x - 2, y - 1] += 1
    if x - 2 >= 0 and y + 1 < 8: table[x - 2, y + 1] += 1
    if x + 2 < 8 and y - 1 >= 0: table[x + 2, y - 1] += 1
    if x + 2 < 8 and y + 1 < 8: table[x + 2, y + 1] += 1
    if y - 2 >= 0 and x - 1 >= 0: table[x - 1, y - 2] += 1
    if y - 2 >= 0 and x + 1 < 8: table[x + 1, y - 2] += 1
    if y + 2 < 8 and x - 1 >= 0: table[x - 1, y + 2] += 1
    if y + 2 < 8 and x + 1 < 8: table[x + 1, y + 2] += 1
    return table


def move4(table, x, y, s):
    if s == 0: s = -1
    if 0 <= x + s < 8 and 0 <= y + s < 8 and table[x + s, y + s] >= 20: table[x + s, y + s] += 1
    if 0 <= x + s < 8 and 0 <= y - s < 8 and table[x + s, y - s] >= 20: table[x + s, y - s] += 1
    return table


def move5(table, x, y):
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= x + i < 8 and 0 <= y + j < 8 and abs(i) + abs(j) != 0:
                table[x + i, y + j] += 1
    return table


def possible_moves(moves, x, y, m, p):
    for move in m:
        if move == 1: moves = move1(moves, x, y)
        if move == 2: moves = move2(moves, x, y)
        if move == 3: moves = move3(moves, x, y)
        if move == 4:
            # print(fig, table[kx, ky], ord(fig) - ord(fig.lower()) == 0)
            moves = move4(moves, x, y, p)
        if move == 5: moves = move5(moves, x, y)
    return moves


def make_table(table, kx, ky):
    moves = np.zeros((8, 8), dtype=int)
    moves[kx, ky] = 100
    for fig in figs:
        fig_locs = np.where(table == fig)
        fig_locs = np.array([fig_locs[:][0], fig_locs[:][1]]).transpose()
        # print(fig)
        for fig_loc in fig_locs:
            x, y = fig_loc[0], fig_loc[1]
            if fig != table[kx, ky]:
                moves[x, y] = 20

    return moves


def check(table, k):
    f = 0
    kx, ky = k[0], k[1]
    check_figs = []
    new_table = make_table(table, kx, ky)
    moves = make_table(table, kx, ky)

    for fig in figs:
        fig_locs = np.where(table == fig)
        fig_locs = np.array([fig_locs[:][0], fig_locs[:][1]]).transpose()
        for fig_loc in fig_locs:
            new_table[kx, ky] = 100
            moves = new_table.copy()

            x, y = fig_loc[0], fig_loc[1]
            p = ord(fig) - ord(fig.lower()) != ord(table[kx, ky]) - ord(table[kx, ky].lower())

            if p and table[kx, ky] != table[x, y]:
                # print(fig, table[kx, ky])
                moves = possible_moves(moves, x, y, figs[fig][1], ord(fig) - ord(fig.lower()) == 0)

                if moves[kx, ky] >= 101:
                    c = check_move(moves, k, x, y, fig)
                    # if fig == 'q' and table[kx, ky] == 'P': print(moves, c)
                    if c:
                        f = 1
                        check_figs.append(fig_loc)

    return f, check_figs
    # print(new_table)


def checkmate(table, king, p, check_figs):
    x, y = king[0], king[1]
    # print(check_figs)
    cm = 1
    # print(check_figs)
    for figure in check_figs:
        # print(dx, dy, x, y, figure, table[x, y], table[figure[0], figure[1]])
        for possfig in figs:
            fig_locs = np.where(table == possfig)
            fig_locs = np.array([fig_locs[:][0], fig_locs[:][1]]).transpose()
            # print(fig_locs)
            for defend in fig_locs:
                defx, defy = defend[0], defend[1]
                moves = make_table(table, x, y)
                figure_letter = table[figure[0], figure[1]]
                moves = possible_moves(moves, figure[0], figure[1], figs[figure_letter][1], ord(figure_letter) - ord(figure_letter.lower()) == 0)
                defend_letter = table[defend[0], defend[1]]
                p = ord(defend_letter) - ord(defend_letter.lower()) == ord(table[x, y]) - ord(table[x, y].lower())
                # print(defend_letter)
                if p and table[x, y] != defend_letter:
                    moves = possible_moves(moves, defx, defy, figs[defend_letter][1], ord(defend_letter) - ord(defend_letter.lower()) == 0)
                    num_overlaps = np.where(moves == 2)
                    num_overlaps = np.array([num_overlaps[0], num_overlaps[1]]).transpose()
                    for overlap in num_overlaps:
                        temp = table.copy()
                        temp[overlap[0], overlap[1]] = table[defx, defy]
                        temp[defx, defy] = '0'
                        if not check(temp, [x, y])[0] and check_move(moves, [overlap[0], overlap[1]], defx, defy, defend_letter): return 0

        c, new_figs = check(table, figure)
        if c:
            for nf in new_figs:
                # print(nf)
                if x != nf[0] or y != nf[1]:
                    temp = table.copy()
                    temp[figure[0], figure[1]] = table[nf[0], nf[1]]
                    temp[nf[0], nf[1]] = '0'
                    if not check(temp, [x, y])[0]: return 0
                else:
                    dx = x - figure[0]
                    dy = y - figure[1]
                    if dx in [-1, 0, 1] and dy in [-1, 0, 1]:
                        temp = table.copy()
                        temp[x - dx, y - dy] = table[x, y]
                        temp[x, y] = '0'
                        # print(check(temp, [x - dx, y - dy])[0])
                        if not check(temp, [x - dx, y - dy])[0]:
                            return 0

    cm = check_king_move(table, x, y)
    return cm

def check_king_move(table, x, y):
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            new = table.copy()
            if 0 <= x + i < 8 and 0 <= y + j < 8 and abs(i) + abs(j) != 0 and new[x + i, y + j] == '0':
                new[x + i, y + j] = new[x, y]
                new[x, y] = '0'
                c = check(new, [x + i, y + j])[0]
                # print(x+i, y+j, c)
                if c == 0: return 0
    return 1


def check_move(moves, k, x, y, fig):
    # if fig == 'q' and table[k[0], k[1]] == 'P': print(moves)
    diff_x, diff_y = k[0] - x, k[1] - y

    # if fig == 'q' and table[k[0], k[1]] == 'P': print(diff_x, diff_y)
    sx, sy = np.sign(diff_x), np.sign(diff_y)
    condition_knight = ((np.abs(diff_x) == 1 and np.abs(diff_y) == 2) or (np.abs(diff_x) == 2 and np.abs(diff_y) == 1))
    condition_pawn = (np.abs(diff_x) == 1 and np.abs(diff_y) == 1)
    condition_king = (np.abs(diff_x) == 1 or np.abs(diff_y) == 1) and fig.lower() == 'k'
    # print(condition_king)
    # print(fig)
    # print(moves, diff_x, diff_y)
    if np.abs(diff_x) == np.abs(diff_y) and np.abs(diff_x) > 1:
        for i in range(sx, diff_x - sx, sx):
            for j in range(sy, diff_y - sy, sy):
                if abs(i) == abs(j) and moves[x + i, y + j] != 1:
                    # print(i, j)
                    return 0
        return 1
    elif diff_x == 0 and diff_y != 0 and abs(diff_y) >= 1:
        return np.nonzero(moves[x, int(min(y, k[1])) + 1: int(max(y, k[1]))] >= 20)[0].sum() == 0
    elif diff_y == 0 and diff_x != 0 and abs(diff_x) >= 1:
        return np.nonzero(moves[int(min(x, k[0])): int(max(x, k[0])), int(y)] >= 20)[0].sum() == 0
    elif condition_king or condition_pawn or condition_knight:
        return 1
    return 0


# up-down-left-right - 1, diagonally -2, knight - 3, pawn - 4, king - 5
figs = {
    'b': [[210, 12, 648], [2]],
    'k': [[232, 92, 576], [5]],
    'n': [[331, 35, 536], [3]],
    'p': [[230, 0, 670], [4]],
    'q': [[294, 39, 528], [1, 2]],
    'r': [[272, 45, 586], [1]],
    'B': [[154, 102, 648], [2]],
    'K': [[136, 176, 576], [5]],
    'N': [[140, 223, 536], [3]],
    'P': [[76, 162, 670], [4]],
    'Q': [[230, 144, 536], [1, 2]],
    'R': [[128, 134, 586], [1]]
}

sum = 0
wrong = []

for i in range(51):
    image_path = 'private/set/{0}/{0}.png'.format(i)
    output = open('private/outputs/{0}.txt'.format(i), 'r', encoding='UTF8')
    lines = output.readlines()

    image_file = Image.open(image_path)
    image = np.array(image_file)
    p = np.nonzero(image)

    start_pixel, end_pixel = [p[0][0], p[1][0]], [p[0][-1], p[1][-1]]
    cut_image = image[start_pixel[0]: end_pixel[0] + 1, start_pixel[1]: end_pixel[1] + 1]
    top_left = str(start_pixel[0]) + ',' + str(start_pixel[1])

    tile_width = len(cut_image[0]) // 8
    tile_length = len(cut_image) // 8
    string = ""
    table = np.zeros((8, 8), dtype=str)

    for row in range(8):
        string_row = ""
        count = 0
        for column in range(8):
            tile = cut_image[row * tile_length: (row + 1) * tile_length, column * tile_width: (column + 1) * tile_width]
            if np.average(tile) == np.average(tile[0, 0, :]):
                count += 1
                table[row, column] = '0'
            else:
                if count != 0:
                    string_row += str(count)
                    count = 0
                fig = detect_piece(tile, figs)
                table[row, column] = fig
                string_row += fig
        if count != 0:
            string_row += str(count)
        string += string_row + "/"
    # print(table)
    ch = '-'
    cm = 0
    k = np.where(table == 'k')
    black_king = [k[0][0], k[1][0]]
    cb, check_figs = check(table, black_king)
    if cb:
        ch = 'W'
        cm = checkmate(table, black_king, 1, check_figs)
    k = np.where(table == 'K')
    white_king = [k[0][0], k[1][0]]
    cw, check_figs = check(table, white_king)
    # print(cb)
    if not cb and cw:
        ch = 'B'
        cm = checkmate(table, white_king, 0, check_figs)
    # print(top_left)
    # print(string[:-1] + '\n' == lines[1])
    print(str(ch) + '\n' == lines[2], cm == int(lines[3]))
    # print(cm == int(lines[3]))
    if cm != int(lines[3]):
        sum += 1
        wrong.append(i)

print(sum, wrong)
