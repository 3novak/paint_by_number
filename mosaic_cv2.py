
# paint by number
# turn a picture into a bunch of colored numbers corresponding to the hue values

import cv2
import numpy as np


def rgb_to_hex(rgb, output='abbreviated'):
    rgb = (rgb[0], rgb[1], rgb[2])
    tmp = '#%02x%02x%02x' % rgb
    if output == 'string':
        return tmp
    elif output == 'abbreviated':
        return ''.join([tmp[1], tmp[3], tmp[5]]).upper()


def avg_rgb(list_rgb):
    # list_rgb is an array of np arrays containing rgb values
    # find the means of the ith components in the vectors contained in the block
    new_rgb = []
    for i in range(0, len(list_rgb)):
        tmp_avg = np.array([vector[i] for block in list_rgb for vector in block]).mean()
        new_rgb.append(tmp_avg)
    print('here it is:', new_rgb)
    new_rgb = np.array(new_rgb).round()
    return new_rgb


def avg_rgb_vec(np_block):
    # accommodates standard numpy arrays
    np_block = np_block.reshape(-1, np_block.shape[-1])
    new_rgb = []
    for i in range(0, 3):
        tmp_value = np.array([vector[i] for vector in np_block]).mean().round()
        new_rgb.append(tmp_value)
    return new_rgb


# get_block() returns a block of size mxm except where start_row and start_col
# are too large for a given m value to accommodate square blocks
def get_block(array, m, start_row, start_col):
    block = [[x for x in row[start_col:start_col+m]] for row in array[start_row:start_row+m]]
    return np.array(block)


# yields blocks of size mxm (except perhaps on edges) via the get_block()
# function given a numpy array
def click_through(array, m=10):
    h = array.shape[0]
    w = array.shape[1]
    for row_ix in range(0, h, m):
        for col_ix in range(0, w, m):
            yield get_block(array, m, start_row=row_ix, start_col=col_ix)


if __name__ == '__main__':
    im = cv2.imread('/Users/ewnovak/Downloads/asdf.jpg')
    font = cv2.FONT_HERSHEY_SIMPLEX
    #COMPLEX_SMALL
    block_size = 10
    r2 = click_through(im, m=block_size)
    num_cols = np.ceil(im.shape[1]/block_size)
    num_rows = np.ceil(im.shape[0]/block_size)
    count_cols = np.floor(im.shape[1]/block_size)
    count_rows = np.floor(im.shape[0]/block_size)
    new_im = np.zeros(im.shape)
    c = 0
    for i in r2:
        print((int((c % num_cols) * new_im.shape[1]/count_cols),
        int((c % num_rows) * new_im.shape[0]/count_rows)))
        cv2.putText(im,
                    rgb_to_hex(avg_rgb_vec(i)),
                    (int((c % num_cols) * new_im.shape[1]/count_cols),
                    int(int(c/count_cols) * new_im.shape[0]/count_rows)),
                    font,
                    1,
                    avg_rgb_vec(i),
                    2)
        c += 1
    cv2.imwrite('wedidit.jpg', im)
    cv2.imshow('im', im)

    # print the 'rough hexadecimal' approximation for the 3x3 block
    # in the following fashion.
    # 1. take the color values of the 3x3 block
    # 2. compute the mean of these 9 pixels
    # 3. return the first base 16 digit as an approximation for each rgb
