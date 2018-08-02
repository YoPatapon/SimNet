import os
import sys
import numpy as np
from tqdm import tqdm

def sort_result(result_file, out_file):
    json_list = list()
    text_list = list()
    score_list = list()
    for line in tqdm(open(result_file, 'r')):
        json, text, score = line.strip().split(',')
        json_list.append(json)
        text_list.append(text)
        score_list.append(float(score))

    order = np.argsort(np.array(score_list))[::-1]
    if os.path.exists(out_file):
        os.remove(out_file)

    with open(out_file, 'a') as f:
        for o in order:
            f.write('{},{},{}\n'.format(json_list[o], text_list[o], score_list[o]))

if __name__ == '__main__':
    sort_result(sys.argv[1], sys.argv[2])
