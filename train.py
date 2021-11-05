import argparse
from utils import *
from LINE_Model import LINEModel
from tqdm import trange
import torch
import torch.optim as optim
import sys
from sklearn import  cluster
import pickle

# 1.设置模型参数
# 2.读图，存点和边并做归一化3.计算点和边的alias table
# 4.Line模型实现
# 5.模型按边训练以及负采样
# 6.结果展示和可视化
if __name__ == "__main__":
    # 1. # 设置模型参数；读图，存点和边并做归一化
    # 1）设置模型参数设置模型超参数，如1st order，2nd order，负样本数量（K），embedding维度，batch、epoch、learning rate等
    # 2）输入输出
    # 输入文件./data/weighted.karate.edgelist
    # 输出文件./model.pt
    parser = argparse.ArgumentParser()
    # 输入文件
    parser.add_argument("-g", "--graph_path", type=str, default='./data/data02/weighted.karate.edgelist')
    # 模型信息输出文件
    parser.add_argument("-save", "--save_path", type=str, default='./saved_model/model.pt')
    # 模型损失函数值输出文件
    parser.add_argument("-lossdata", "--lossdata_path", type=str, default='./saved_model/loss.pkl')

    # Hyperparams.超参数
    # 论文中的1st order，2nd order
    parser.add_argument("-order", "--order", type=int, default=2)
    # 负样本个数
    parser.add_argument("-neg", "--negsamplesize", type=int, default=5)
    # embedding维度
    parser.add_argument("-dim", "--dimension", type=int, default=128)
    # batchsize大小
    parser.add_argument("-batchsize", "--batchsize", type=int, default=5)
    parser.add_argument("-epochs", "--epochs", type=int, default=1)
    # 学习率
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.025)  # As starting value in paper
    # 负采样指数值
    parser.add_argument("-negpow", "--negativepower", type=float, default=0.75)
    args = parser.parse_args()

    # 2.读图，存点和边并做归一化
    # 1）读图自己实现的makeDist函数，在utils.py中
    # Create dict of distribution when opening file
    # 读图，函数在utils.py中
    edgedistdict, nodedistdict, weights, nodedegrees, maxindex = makeDist(args.graph_path, args.negativepower)

    # 3. 计算点和边的alias table
    # 构建alias table，达到O(1)的采样效率
    edgesaliassampler = VoseAlias(edgedistdict)
    nodesaliassampler = VoseAlias(nodedistdict)

    # LINE模型实现
    # 每次训练batch size大小的边数量
    batchrange = int(len(edgedistdict) / args.batchsize)
    print('maxindex = ', maxindex)
    line = LINEModel(maxindex + 1, embed_dim=args.dimension, order=args.order)
    # SGD优化，nesterov是对momentum的改进，是在momentum向量的终端再更新梯度。
    opt = optim.SGD(line.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    # 选用gpu或cpu训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lossdata = {"it": [], "loss": []}
    it = 0

    print("\nTraining on {}...\n".format(device))
    # 共训练epoch次数
    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch))
        # 每次训练组数：batchsize
        for b in trange(batchrange):
            # edgesaliassampler是实现alias building的VoseAlias类，这里采样出batchsize条边
            samplededges = edgesaliassampler.sample_n(args.batchsize)
            # 存makeData是utils.py中的函数，为每条边采样出K条负样本边存每一条格式是（node i，node j，negative nodes...）
            batch = list(makeData(samplededges, args.negsamplesize, weights, nodedegrees, nodesaliassampler))
            # 转换成tensor格式
            batch = torch.LongTensor(batch)
            # 把一个batch的数据打印出来是这样：
            # tensor([[3, 8 14, 14, 24, 2, 32],
            #         [25, 32, 14, 9, 4, 24, 23],
            #         [1, 14, 32, 1, 25, 27, 16],
            #         [26, 32, 30, 4, 14, 7, 4],
            #         [25, 32, 25, 14, 20, 14, 27]])

            # 取第0列就是起始点
            v_i = batch[:, 0]
            # 取第1列就是终点
            v_j = batch[:, 1]
            # 取后面5列就是负样本
            negsamples = batch[:, 2:]
            # 在做BP之前将gradients置因为是梯度累加的
            line.zero_grad()
            # Line模型实现部分
            loss = line(v_i, v_j, negsamples, device)
            # 计算梯度
            loss.backward()
            # 根据梯度值更新参数值
            opt.step()

            lossdata["loss"].append(loss.item())
            lossdata["it"].append(it)
            it += 1

    print("\nDone training, saving model to {}".format(args.save_path))
    torch.save(line, "{}".format(args.save_path))

    print("Saving loss data at {}".format(args.lossdata_path))
    with open(args.lossdata_path, "wb") as ldata:
        pickle.dump(lossdata, ldata)
    # sys.exit()

    # k-means
    embedding_node = []
    for i in range(1, 35):
        i = torch.LongTensor([i])
        t = line.nodes_embeddings(i)
        embedding_node.append(t.tolist()[0])
    embedding_node = np.matrix(embedding_node).reshape((34, -1))
    y_pred = cluster.KMeans(n_clusters=3, random_state=9).fit_predict(embedding_node)  # 调用 test_RandomForestClassifier
    print('y_pred = ', y_pred)

