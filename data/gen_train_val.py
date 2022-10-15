import os
import random


def gen_train_val(path, ratio):
    imgs = os.listdir(path)
    train_num = int(len(imgs) * ratio)
    train_imgs = random.sample(imgs, train_num)
    val_imgs = []
    for img in imgs:
        if img not in train_imgs:
            val_imgs.append(img)
    with open('./pretrain.txt', 'a', encoding='utf-8') as f:
        for train_img in train_imgs:
            f.write(os.path.dirname(os.path.abspath(__file__))+'/pretrain/images/'+train_img+'\n')
            # f.write(train_img.split('.')[0]+ '\n')
    with open('./preval.txt', 'a', encoding='utf-8') as f:
        for val_img in val_imgs:
            f.write(os.path.dirname(os.path.abspath(__file__))+'/pretrain/images/'+val_img+'\n')
            # f.write(val_img.split('.')[0]+'\n')


if __name__ == '__main__':
    path = './pretrain/images'
    ratio = 0.9
    gen_train_val(path, ratio)
