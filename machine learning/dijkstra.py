import sys

# 3个数一组，例如1，2，1表示1到2的有向距离为1
graph = [1, 2, 1, 1, 3, 12, 2, 3, 9, 2, 4, 3, 3, 5, 5, 4, 3, 4, 4, 5, 13, 4, 6, 15, 5, 6, 4]
# 图的定点数
points = 6
# 图的边数
sides = 9
# 邻接矩阵
e = [[i for j in range(points)] for i in range(points)]
# 最短距离
dis = [i for i in range(points)]
# 定点是否已经访问 0-表示没有访问 1-表示已经访问
book = [0 for i in range(points)]
# 定义最大值
MAX = sys.maxsize
# 定义最小值
MIN = sys.maxsize
# 邻接矩阵初始化
for i in range(points):
    for j in range(points):
        if i == j:
            e[i][j] = 0
        else:
            e[i][j] = MAX
# 对邻接矩阵赋值
for i in range(0, len(graph), 3):
    e[graph[i] - 1][graph[i + 1] - 1] = graph[i + 2]
# 这里是获取第一个定点到其余各个定点的距离
dis = e[0]
# 标记第一个定点已经确定
book[0] = 1
u = None
# 下面是核心算法
for i in range(points - 1):
    # 注意这里的MIN循环一定要重新赋值
    MIN = MAX
    # 寻找到没有确定的定点的距离最小的点
    for j in range(points):
        if book[j] == 0 and dis[j] < MIN:
            MIN = dis[j]
            u = j
    # 并赋值1，表示已经经过
    book[u] = 1

    for j in range(points):
        if e[u][j] < MAX and dis[j] > dis[u] + e[u][j]:
            dis[j] = dis[u] + e[u][j]

print(dis)
