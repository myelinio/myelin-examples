import numpy as np
from PIL import Image, ImageDraw


def test():
    with Image.open('demo/72828977-189686_CWK180204_FLP_01_0000_max_600x600.jpg') as im:
        # draw on the origin img
        draw = ImageDraw.Draw(im)
        with open('demo/72828977-189686_CWK180204_FLP_01_0000_max_600x600.txt', 'r') as f:
            anno_list = f.readlines()
        for anno in anno_list:
            anno_colums = anno.strip().split(',')
            anno_array = np.array(anno_colums)[:-1]
            xy_list = np.reshape(anno_array.astype(float), (4, 2))
            draw.line([tuple(xy_list[0]), tuple(xy_list[1]), tuple(xy_list[2]),
                       tuple(xy_list[3]), tuple(xy_list[0])],
                      width=1,
                      fill='red')
        im.save('demo/LB1xbbUGVXXXXaIXFXXXXXXXXXX_anno.jpg')


test()
