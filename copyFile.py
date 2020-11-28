import os
import shutil


root = 'D:\zjw\Linux\CT'
for i in range(1, 299):
# —————————————————————————————————— 剪切文件 begin ——————————————————————————————————
    sourceDir = root + '/' + str(i)
    list = os.listdir(sourceDir)
    targetDir = sourceDir + '/0'
    for files in list:
        sourceFile = os.path.join(sourceDir, files)
        targetFile = os.path.join(targetDir, files)
        try:
            if os.path.isfile(sourceFile):
                # 可以试试不加>0的后果
                shutil.move(sourceFile, targetFile)
        except FileNotFoundError:
            print("此文件夹不存在，请重新检查！")
# —————————————————————————————————— 剪切文件 end ——————————————————————————————————
    # for k in range(0, 1):
    #     path2 = path + '/' + str(k)
    #     os.makedirs(path2)