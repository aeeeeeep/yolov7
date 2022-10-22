import os

def rename():
    i = 0
    path = './train/images'

    filelist = os.listdir(path)
    for files in filelist:
        i = i + 1
        Olddir = os.path.join(path, files)
        if os.path.isdir(Olddir):
            continue
        filename = files[:3]
        filetype = '.png'
        Newdir = os.path.join(path, filename + filetype)
        os.rename(Olddir, Newdir)
    return True

if __name__ == '__main__':
    rename()
