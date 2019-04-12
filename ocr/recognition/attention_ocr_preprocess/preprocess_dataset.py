import argparse
import glob
import os
import random
import string
import subprocess

import numpy as np
from PIL import Image
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def createDataset(outputPath, imagePathList, labelList, stop_words_per):
    '''
    aocr dataset ./datasets/annotations-training.txt ./datasets/training.tfrecords
    datasets/images/world.jpg world
    '''
    print(len(imagePathList), len(labelList))
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    stop_words = set(stopwords.words('english'))
    outputFile = open(outputPath, 'w')
    widths = []
    heights = []
    for i in range(nSamples):
        # im = Image.open(imagePathList[i])
        # width, height = im.size
        # widths.append(width)
        # heights.append(height)
        if (labelList[i].lower() not in stop_words) or (random.random() < stop_words_per):
            image_path = imagePathList[i]
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            dirname = os.path.dirname(image_path)

            # # preprocess
            # processed_base_path = os.path.join(dirname, "processed")
            # if not os.path.exists(processed_base_path):
            #     os.makedirs(processed_base_path)
            # processed_path = os.path.join(processed_base_path, "%s.png" % filename)
            # subprocess.call(["/home/ryadh/Dev/tools/image_proc/imagemagick/textcleaner", "-g",  "-e","normalize", "-f"  ,"30" ,"-o", "12", "-s", "2",
            #                  "-a",
            #                  "10",
            #                  "-g",
            #                  image_path,
            #                  processed_path
            #                  ])
            outputFile.write("%s %s\n" % (image_path, labelList[i]))
        # outputFile.write("%s;%s\n" % (imagePathList[i], tf_crnn_label(labelList[i])))

    # print("heights: ", np.percentile(np.array(heights), [10, 50, 75, 90, 95, 99]))
    # print("widths: ", np.percentile(np.array(widths), [10, 50, 75, 90, 95, 99]))

    outputFile.close()
    print('Created dataset with %d samples' % nSamples)


def read_lines(file_path):
    file = open(file_path)
    lines = []
    for l in file:
        lines.append(l.strip())
    return lines


def filter_non_supported(str, supported_chars):
    return ''.join((filter(lambda x: x in supported_chars, str)))


def key_indx(x):
    return int(digits(x.split('/')[-1].split('_')[-1].split(".")[0]))


def tf_crnn_label(label):
    return ''.join(map(lambda x: x + '|', label))


def digits(strg):
    return ''.join(ch for ch in strg if ch.isdigit())
# od -cvAnone -w1 /media/ryadh/DATA1/Riadh_Data/data/floorplan/crops4/labels_fixed.txt| sort -b | uniq -c | sort -rn
# sed '65930,65940 !d' /media/ryadh/DATA1/Riadh_Data/data/floorplan/crops_white_rm_floorplan/labels_fixed.txt
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputFile', required=True)
    parser.add_argument('--stop-words-per', type=float)

    parser.add_argument('--test', default=False, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--data-paths', action='append', help='data paths', required=True)

    parser.add_argument('--sizes', action='append', help='Dataset sizes', required=True)

    parser.add_argument('--starts',action='append', help='Dataset sizes', required=True)

    args = parser.parse_args()

    assert(len(args.data_paths) == len(args.sizes))
    assert(len(args.data_paths) == len(args.starts))

    labelList_all = []
    imagePathList_all = []
    for i in range(len(args.data_paths)):
        start = int(args.starts[i])
        size = int(args.sizes[i])
        data_path = args.data_paths[i]
        print("Reading from %s start: %s, size: %s" % (data_path, start, size))

        labelList = read_lines(os.path.join(data_path, 'labels_fixed.txt'))
        imagePathList = glob.glob(os.path.join(data_path, '*.jpg'))
        imagePathList = sorted(imagePathList, key=key_indx)

        assert (len(imagePathList) == len(labelList))
        if args.test:
            imagePathList_all.extend(imagePathList[-size:])
            labelList_all.extend(labelList[-size:])
        else:
            imagePathList_all.extend(imagePathList[start:start+size])
            labelList_all.extend(labelList[start:start+size])

    assert(len(labelList_all) == len(imagePathList_all))


    supported_chars = list(string.ascii_letters) + read_lines('./supported_chars')

    labelListFiltered = [filter_non_supported(x, supported_chars) for x in labelList_all]

    c = list(zip(imagePathList_all, labelListFiltered))
    random.shuffle(c)
    imagePathList, labelListFiltered = zip(*c)

    createDataset(args.outputFile, imagePathList, labelListFiltered, args.stop_words_per)
# aocr dataset ./datasets/annotations-training-white.txt ./datasets/training-white-floorplan.tfrecords
