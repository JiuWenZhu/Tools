import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd

font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}


def plot_trainD_segD():
    x =['Projected', 'Segmented', 'Final']
    ACC= [0.8777, 0.8842, 0.9253]
    VACC = [0.9419, 0.9543, 0.9612]
    AUC = [0.9708, 0.9831, 0.9936]

    plt.plot(x, ACC,label='Mean ACC',linewidth=3,color='b',marker='*', markerfacecolor='blue',markersize=10)
    for a, b in zip(x, ACC):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=16)
    plt.plot(x, VACC, label='Mean VACC', linewidth=3, color='r', marker='*', markerfacecolor='Red', markersize=10)
    for a, b in zip(x, VACC):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=16)
    plt.plot(x, AUC, label='Mean AUC', linewidth=3, color='g', marker='*', markerfacecolor='Green', markersize=10)
    for a, b in zip(x, AUC):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=16)
    plt.xlabel('Different stages of data.', font1)
    plt.legend()
    plt.show()


def box_plot():
    def single(sheet_name):
        data = pd.read_excel('D:/1 data/CT/plot_ROC/box_plot.xlsx', sheet_name=sheet_name)
        vgg, resnet34, pointnet, ssg, unun, proposed = data['vgg'], data['resnet34'], data['pointnet'], data['ssg'], \
                                                       data['unun'], data['proposed']
        plt.figure(figsize=(7.5, 5))  # 设置画布的尺寸
        plt.title(sheet_name, fontsize=20)  # 标题，并设定字号大小
        labels = '3D VGG16', '3D ResNet34', 'PointNet', 'PointNet++', 'Proposed', 'Proposed (Pretrained)'
        plt.boxplot([vgg, resnet34, pointnet, ssg, unun, proposed], labels=labels)  # grid=False：代表不显示背景中的网格线
        plt.show()
    single('ACC')
    single('VACC')
    single('AUC')


def plot_multi_scale_ROC():
    def plot_ROC(scale):
        root = 'D:/1 data/CT/plot_ROC/'
        def loadfprtpr(path):
            model_path = root + path
            fpr = np.loadtxt(model_path + '_fpr.txt')
            tpr = np.loadtxt(model_path + '_tpr.txt')
            auc = metrics.auc(fpr, tpr)
            return fpr, tpr, auc
        vgg_fpr, vgg_tpr, auc = loadfprtpr('/vgg')
        plt.plot(vgg_fpr, vgg_tpr, lw=1, color='yellow', label='3D VGG16 with cropping and rotation')
        resnet34_fpr, resnet34_tpr, auc = loadfprtpr('/resnet34')
        plt.plot(resnet34_fpr, resnet34_tpr, lw=1, color='green', label='3D ResNet34 with cropping and rotation')
        pointnet_fpr, pointnet_tpr, auc = loadfprtpr('/pointnet')
        plt.plot(pointnet_fpr, pointnet_tpr, lw=1, color='orange', label='PointNet (Pretrained)')
        ssg_fpr, ssg_tpr, auc = loadfprtpr('/ssg')
        plt.plot(ssg_fpr, ssg_tpr, lw=1, color='pink', label='PointNet++ (Pretrained)')
        unun_fpr, unun_tpr, auc = loadfprtpr('/unun')
        plt.plot(unun_fpr, unun_tpr, lw=1, color='red', label='Proposed')
        proposed_fpr, proposed_tpr, auc = loadfprtpr('/proposed')
        plt.plot(proposed_fpr, proposed_tpr, lw=1, color='blue', label='Proposed (Pretrained)')
        plt.xlim(scale)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend()
        plt.show()
    plot_ROC([-0.0001, 0.051])
    # plot_ROC([-0.0001, 1.0001])


def plot_bar():
    # ######################## plot loss function #####################################
    name_list = ['ArcFace', 'AM-softmax', 'Center loss', 'Proposed']
    ACC = [0.7061, 0.7633, 0.482, 0.8604]
    VACC = [0.8314, 0.873, 0.7314, 0.9279]
    AUC = [0.8949, 0.9438, 0.8099, 0.9815]
    x = list(range(len(name_list)))
    total_width, n = 0.8, 3
    width = total_width / n
    plt.figure(figsize=[10, 5])
    for i in range(len(x)):
        x[i] -= width
    plt.bar(x, ACC, width=width, label='ACC', color='deepskyblue')
    for a, b in zip(x, ACC):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=14)

    for i in range(len(x)):
        x[i] += width
    plt.bar(x, VACC, width=width, label='VACC', tick_label=name_list, color='pink')
    for a, b in zip(x, VACC):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=14)

    for i in range(len(x)):
        x[i] += width
    plt.bar(x, AUC, width=width, label='AUC', tick_label=name_list, fc='green')
    for a, b in zip(x, AUC):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=14)
    # plt.xticks(fontproperties='Times New Roman', fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Results of different loss function by using CT depth images.',
              fontdict={'family':'Times New Roman', 'size':16})
    plt.legend(prop={'family': 'Times New Roman', 'size': 15}, loc='lower right')
    plt.show()

    # ######################## plot loss function + pretrain #####################################
    ACC = [0.6318, 0.7743, 0.6353, 0.9123]
    VACC = [0.6903, 0.8888, 0.7536, 0.9533]
    AUC = [0.7418, 0.9522, 0.8396, 0.992]
    x = list(range(len(name_list)))
    total_width, n = 0.8, 3
    width = total_width / n
    plt.figure(figsize=[10, 5])
    for i in range(len(x)):
        x[i] -= width
    plt.bar(x, ACC, width=width, label='ACC', color='deepskyblue')
    for a, b in zip(x, ACC):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=14)

    for i in range(len(x)):
        x[i] += width
    plt.bar(x, VACC, width=width, label='VACC', tick_label=name_list, color='pink')
    for a, b in zip(x, VACC):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=14)

    for i in range(len(x)):
        x[i] += width
    plt.bar(x, AUC, width=width, label='AUC', tick_label=name_list, fc='green')
    for a, b in zip(x, AUC):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=14)
    # plt.xticks(fontproperties='Times New Roman', fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Pretrained results of different loss function by using CT depth images.',
              fontdict={'family': 'Times New Roman', 'size': 16})
    plt.legend(prop={'family': 'Times New Roman', 'size': 15}, loc='lower right')
    plt.show()

# plot_trainD_segD()
# box_plot()
# plot_multi_scale_ROC()
plot_bar()