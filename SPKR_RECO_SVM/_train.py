from sklearn import svm
import read_data


train_file = ""
pred_file = ""


def train(infile, c=1, kernel='linear'):
    x, y = read_data.get_input(infile, 'train')
    model = svm.SVC(C=c, kernel=kernel)
    model.fit(x, y)
    return model


def pred(infile, model):
    support = 0
    data_num = 0
    ceps, label = read_data.get_input(infile, 'predict')
    for index in range(ceps):
        data_num += 1
        spkr_id = model.predict(ceps[index])
        if spkr_id == label[index]:
            support += 1
    accurancy = float(support/data_num)
    return accurancy


def main():
    model = train(train_file, 1, 'linear')
    print(pred(pred_file, model))


if __name__ == '__main__':
    main()
