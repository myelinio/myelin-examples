import argparse
import json
import os

import sys
from PIL import Image


def copy_img(base_f_name, input_dir, output_dir):
    im = [f for f in os.listdir(input_dir) if '.json' not in f and base_f_name in f][0]
    target = os.path.join(output_dir, base_f_name + '.jpg')
    Image.open(os.path.join(input_dir, im)).convert('RGB').save(target)


def write_line(file, txt):
    file.write(txt + '\n')
    file.flush()


def process_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')

    parameters = parser.parse_args(args)
    return parameters


def main(args):
    args = process_args(args)
    labelled = [f for f in os.listdir(args.input_dir) if '.json' in f]
    for f_d in labelled:
        with open(os.path.join(args.input_dir, f_d)) as f:
            base_f_name = f_d.split('.')[0]
            copy_img(base_f_name)
            data = json.load(f)
            txt_file = open(os.path.join(args.output_dir, base_f_name + '.txt'), 'w')

            for poly in data['shapes']:
                pts = poly['points']
                pts_a = [pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1],
                         'LABEL']
                write_line(txt_file, ",".join([str(x) for x in pts_a]))
            txt_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
