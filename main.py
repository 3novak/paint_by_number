
# paint by number
# turn a picture into a bunch of colored numbers corresponding to the hue values

from PIL import Image
import numpy as np


def get_array(img_location):
    pillow = Image.open(img_location)
    return np.array(pillow)


def rgb_to_hex(rgb, output='string'):
    rgb = (rgb[0], rgb[1], rgb[2])
    tmp = '#%02x%02x%02x' % rgb
    if output == 'string':
        return tmp
    elif output == 'truncated':
        return ' '.join([tmp[1], tmp[3], tmp[5]]).upper()


def avg_rgb(list_rgb):
    # list_rgb is an array of np arrays containing rgb values
    # find the means of the ith components in the vectors contained in the block
    new_rgb = []
    for i in range(0, len(list_rgb)):
        tmp_avg = np.array([vector[i] for block in list_rgb for vector in block]).mean()
        print(tmp_avg)
        new_rgb.append(tmp_avg)
    return np.array(new_rgb).round()


# get_block() returns a block of size mxm except where start_row and start_col
# are too large for a given m value to accommodate square blocks
def get_block(array, m, start_row, start_col):
    block = [[x for x in row[start_col:start_col+m]] for row in array[start_row:start_row+m]]
    return np.array(block)


# yields blocks of size mxm (except perhaps on edges) via the get_block()
# function given a numpy array
def click_through(array, m):
    h = array.shape[0]
    w = array.shape[1]
    for row_ix in range(0, h, m):
        for col_ix in range(0, w, m):
            yield get_block(array, m, start_row=row_ix, start_col=col_ix)


if __name__ == '__main__':
    print(get_array('/Users/ewnovak/Downloads/asdf.jpg'))

    # print the 'rough hexadecimal' approximation for the 3x3 block
    # in the following fashion.
    # 1. take the color values of the 3x3 block
    # 2. compute the mean of these 9 pixels
    # 3. return the first base 16 digit as an approximation for each rgb
    # 4. copy this 1x3 block three times to cover the original 3x3 block.
    # 5. print this 3x3 block with its corresponding font color
    # 6. ensure that the font and font size is best suited to looking like a block

    # see below a sample end product
    # 3 E 2 F 5 3
    # 3 E 2 F 5 3
    # 3 E 2 F 5 3
