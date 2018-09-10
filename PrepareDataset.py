import os

path2train = 'E:\\Deeplearning\\AlexNet\\101_ObjectCategories'

category = os.listdir(path2train)
print(category)
kind = 0

for c in category:
    c_path = path2train + '\\' + c
    c_files = os.listdir(c_path)
    #每类的图片总数
    item_num = len(c_files)
    print(c + ':' + str(item_num))
    #每类训练样本的个数
    train_num = int(0.8 * item_num)
    #每类测试样本的个数
    test_num = item_num - train_num
    with open('./train.txt', 'a+', encoding='utf-8') as f:
        for i in range(train_num):
            fpath = c_path + '\\' + c_files[i]
            f.write(fpath + ' ' + str(kind) + '\n')
    with open('./val.txt', 'a+', encoding='utf-8') as f:
        for j in range(train_num, item_num):
            fpath = c_path + '\\' + c_files[j]
            f.write(fpath + ' ' + str(kind) + '\n')
    kind += 1

