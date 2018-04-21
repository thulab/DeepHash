import numpy as np
import scipy.io as scio

def pr_curve(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])

    query_num = validation_code.shape[0]
    database_num = database_code.shape[0]

    database_code = np.sign(database_code)
    database_code[database_code == -1] = 0
    database_code = database_code.astype(int)

    validation_code = np.sign(validation_code)
    validation_code[validation_code == -1] = 0
    validation_code = validation_code.astype(int)

    database_labels.astype(np.int)
    validation_labels.astype(np.int)

    WTrue = np.dot(validation_labels, database_labels.T)
    WTrue[WTrue >= 1] = 1
    WTrue[WTrue < 1] = 0
    print(WTrue.shape)
    print(np.max(WTrue))
    print(np.min(WTrue))

    DHat = np.zeros((query_num, database_num))

    for i in range(query_num):
        query = validation_code[i, :]
        query_matrix = np.tile(query, (database_num, 1))

        distance = np.sum(np.absolute(query_matrix - database_code), axis=1)
        DHat[i, :] = distance
        print(i)

    print(DHat.shape)
    print(np.max(DHat))
    print(np.min(DHat))

    mat_dic = dict(
        WTrue=WTrue,
        DHat=DHat
    )
    scio.savemat('./data/data.mat', mat_dic)


def precision_recall(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)
    database_labels.astype(np.int)
    validation_labels.astype(np.int)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    ones = np.ones((ids.shape[0], ids.shape[1]), dtype=np.int)
    print(np.min(ids))
    ids = ids + ones
    print(np.min(ids))
    mat_ids = dict(
        ids=ids,
        LBase=database_labels,
        LTest=validation_labels
    )
    scio.savemat('./data/data.mat', mat_ids)


def hamming_precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    database_num = database_code.shape[0]

    database_code = np.sign(database_code)
    database_code[database_code == -1] = 0
    database_code = database_code.astype(int)

    validation_code = np.sign(validation_code)
    validation_code[validation_code == -1] = 0
    validation_code = validation_code.astype(int)

    APx = []

    for i in range(query_num):
        query = validation_code[i, :]
        query_matrix = np.tile(query, (database_num, 1))

        label = validation_labels[i, :]
        label[label == 0] = -1
        label_matrix = np.tile(label, (database_num, 1))

        distance = np.sum(np.absolute(query_matrix - database_code), axis=1)
        similarity = np.sum(database_labels == label_matrix, axis=1)
        similarity[similarity > 1] = 1

        total_rel_num = np.sum(distance <= R)
        true_positive = np.sum((distance <= R) * similarity)

        print('--------')
        print(i)
        print(true_positive)
        print(total_rel_num)
        print('--------')
        if total_rel_num != 0:
            APx.append(float(true_positive) / total_rel_num)
        else:
            APx.append(float(0))

    print(np.sum(np.array(APx) != 0))
    return np.mean(np.array(APx))


def precision_curve(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    query_num = validation_code.shape[0]
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    arr = []

    for iter in range(10):
        R = (iter + 1) * 100
        APx = []
        for i in range(query_num):
            label = validation_labels[i, :]
            label[label == 0] = -1
            idx = ids[:, i]
            imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
            relevant_num = np.sum(imatch)
            APx.append(float(relevant_num) / R)
        arr.append(np.mean(np.array(APx)))
        print(arr)
    print(arr)


def precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        APx.append(float(relevant_num) / R)

    return np.mean(np.array(APx))


def mean_average_precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))


def statistic_prob(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    sim = np.dot(database_code, validation_code.T)
    query_num = validation_code.shape[0]
    database_num = database_code.shape[0]
    ones = np.ones((database_num, query_num))
    exp_sim = np.exp(sim)
    prob = ones / (1 + 1 / exp_sim)
    useless = np.sum(prob >= 0.95) + np.sum(prob <= 0.05)
    useful = query_num * database_num - useless
    print("useful")
    print(useful)
    print("useless")
    print(useless)

