import tarfile
import os 
import shutil

root_dir = '/mnt/lizhaochen'
for image in os.listdir(root_dir):
    ext = image.split('.')[-1]
    if ext=='JPEG':
        print(ext)
        shutil.move(os.path.join(root_dir, image), os.path.join(root_dir, 'val'))

    # folder = tar_file.split('.')[0]
    # print(folder)
    # tar = tarfile.open(os.path.join(root_dir, tar_file))
    # tar.extractall(os.path.join(root_dir,folder))
    # tar.close()
    # os.remove(os.path.join(root_dir, tar_file))

#     if tar_file[0] == 'n':
#         print(tar_file)
#         tar_file = os.path.join(root_dir, tar_file)
#         shutil.move(tar_file, os.path.join(root_dir, 'train'))
# tar_path = "/mnt/lizhaochen/n01773797.tar"

# tar_file = tarfile.open(tar_path)
# tar_file.extractall('./myfolder')
# tar_file.close