from sklearn.semi_supervised import LabelPropagation
import load_data


def define_model():

    model = LabelPropagation()

    return model

def get_acc(fake,true):
    acc = 0


    for i in range(len(fake)):
        if (fake[i] == true[i]):
            acc = acc + 1
    acc = acc / len(fake)

    return acc

def data_process():
    data = load_data.load()[0]
    label = load_data.load()[1]

    train_size = int(0.8*data.shape[0])
    train_data = data[:train_size,:]
    test_data = data[train_size:, :]
    train_label = label[:train_size,:]
    test_label = label[train_size:,:]

    return train_data,train_label,test_data,test_label


def train():
    train_data, train_label, test_data, test_label = data_process()
    model = define_model()
    model.fit(train_data,train_label)

    fake_label_test = model.predict(test_data)
    fake_label_train = model.predict(train_data)

    return get_acc(fake_label_test,test_label),get_acc(fake_label_train,train_label)


result = train()
print("acc in testing set = "+ str(result[0]))
print("acc in training set = "+ str(result[1]))