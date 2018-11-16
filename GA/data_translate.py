import pandas as pd


def arff_to_csv(fpath):
    # 读取arff数据
    if fpath.find('.arff') < 0:
        print('the file is nott .arff file')
        return
    f = open(fpath)
    lines = f.readlines()
    content = []
    for l in lines:
        content.append(l)
    datas = []
    for c in content:
        cs = c.split(',')
        datas.append(cs)

    # 将数据存入csv文件中
    df = pd.DataFrame(data=datas, index=None, columns=None)
    filename = fpath[:fpath.find('.arff')] + '.csv'
    df.to_csv(filename, index=None)


if __name__ == '__main__':
    arff_to_csv(r'D:\tensorflow_learning\GA\ALL-AML_train.arff')
