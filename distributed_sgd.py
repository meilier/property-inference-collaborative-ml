import os
import math
os.environ.setdefault('PATH', '')
import sys
import time
import lasagne
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict

from split_data import prepare_data_biased
from load_lfw import load_lfw_with_attrs, BINARY_ATTRS, MULTI_ATTRS

SAVE_DIR = './grads/'

# 参数的计算可通过global_params来计算
net_victim = 0
net_attacker = 0

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


def build_cnn_feat_extractor(input_var=None, input_shape=(None, 3, 50, 50), n=128):
    assert isinstance(n, int)
    network = OrderedDict()
    network['input'] = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network['conv1'] = lasagne.layers.Conv2DLayer(
        network['input'], num_filters=32, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network['pool1'] = lasagne.layers.MaxPool2DLayer(network['conv1'], pool_size=(2, 2))

    network['conv2'] = lasagne.layers.Conv2DLayer(
        network['pool1'], num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)
    network['pool2'] = lasagne.layers.MaxPool2DLayer(network['conv2'], pool_size=(2, 2))

    network['conv3'] = lasagne.layers.Conv2DLayer(
        network['pool2'], num_filters=n, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)
    network['pool3'] = lasagne.layers.MaxPool2DLayer(network['conv3'], pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network['fc1'] = lasagne.layers.DenseLayer(
        network['pool3'],
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    return network


def build_mt_cnn(input_var=None, classes=2, infer_classes=2, input_shape=(None, 3, 50, 50), n=128):
    network = build_cnn_feat_extractor(input_var, input_shape, n)
    network['fc2'] = lasagne.layers.DenseLayer(
        network['fc1'],
        num_units=classes,
        nonlinearity=lasagne.nonlinearities.linear)

    network['fc2_B'] = lasagne.layers.DenseLayer(
        network['fc1'],
        num_units=infer_classes,
        nonlinearity=lasagne.nonlinearities.linear)

    return network


def build_cnn(input_var=None, classes=2, input_shape=(None, 3, 50, 50), n=128):
    network = build_cnn_feat_extractor(input_var, input_shape, n)
    network['fc2'] = lasagne.layers.DenseLayer(
        network['fc1'],
        num_units=classes,
        nonlinearity=lasagne.nonlinearities.linear)
    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, targets_B=None):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if targets_B is None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt], targets_B[excerpt]


def train_lfw(task='gender', attr="race", prop_id=2, p_prop=0.5, n_workers=2, num_iteration=3000,
              alpha_B=0., victim_all_nonprop=False, balance=False, k=5, train_size=0.3):

    x, y, prop = load_lfw_with_attrs(task, attr)
    prop_dict = MULTI_ATTRS[attr] if attr in MULTI_ATTRS else BINARY_ATTRS[attr]

    print('Training {} and infering {} property {} with {} data'.format(task, attr, prop_dict[prop_id], len(x)))

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    prop = np.asarray(prop, dtype=np.int32)

    indices = np.arange(len(x))
    # 获得race是black的索引,len(prop_indices) = 559
    prop_indices = indices[prop == prop_id]
    # 其余的是asian 和 white
    nonprop_indices = indices[prop != prop_id]

    # 把是black的标记为1，不是black的标记为0
    prop[prop_indices] = 1
    prop[nonprop_indices] = 0

    filename = "lfw_psMT_{}_{}_{}_alpha{}_k{}".format(task, attr, prop_id, alpha_B, k)

    if n_workers > 2:
        filename += '_n{}'.format(n_workers)

    # train_multi_task_ps((x, y, prop), input_shape=(None, 3, 62, 47), p_prop=p_prop, balance=balance,
    #                     filename=filename, n_workers=n_workers, alpha_B=alpha_B, k=k,
    #                     num_iteration=num_iteration, victim_all_nonprop=victim_all_nonprop,
    #                     train_size=train_size)
    train_multi_task_ps((x, y, prop), input_shape=(None, 3, 62, 47), p_prop=p_prop,
                    filename=filename, n_workers=n_workers, alpha_B=alpha_B, k=k,
                    num_iteration=num_iteration, victim_all_nonprop=victim_all_nonprop,
                    train_size=train_size)


def build_worker_attacker(input_shape, classes=2, infer_classes=2, lr=None, seed=54321, alph_B=0.5):
    lasagne.random.set_rng(np.random.RandomState(seed))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    target_var_B = T.ivector('targetsB')

    network_dict = build_mt_cnn(input_var, classes=classes, infer_classes=infer_classes, input_shape=input_shape)

    network, network_B = network_dict['fc2'], network_dict["fc2_B"]
    prediction, prediction_B = lasagne.layers.get_output([network, network_B])

    prediction = lasagne.nonlinearities.softmax(prediction)
    prediction_B = lasagne.nonlinearities.softmax(prediction_B)

    loss_A = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss_B = lasagne.objectives.categorical_crossentropy(prediction_B, target_var_B)

    loss = (1 - alph_B) * loss_A.mean() + alph_B * loss_B.mean()

    # save init
    params = lasagne.layers.get_all_params([network, network_B], trainable=True)

    # 如果设置lr那么就是本地训练后再参数上传
    if lr is not None:
        updates = lasagne.updates.sgd(loss, params, lr)
        train_fn = theano.function([input_var, target_var, target_var_B], loss, updates=updates)
        return network, params, train_fn
    #params: [W, b, W, b, W, b, W, b, W, b, W, b]
    grads = T.grad(loss, params)
    # 取出最后两层 [W, b]
    params_B = params[-2:]
    # 删除最后两层 params:[W, b, W, b, W, b, W, b, W, b]，和其他victim正好相同的网络结构
    params = params[:-2]
    # 提取最后两层
    # grads_B:[dot.0, InplaceDimShuffle{1}.0]
    grads_B = grads[-2:]
    # 删除最后两层
    grads = grads[:-2]

    p_idx = 0
    grads_dict = dict()
    for p, g in zip(params, grads):
        key = p.name + str(p_idx)
        p_idx += 1
        grads_dict[key] = g
    #grads_dict : {'W0': AbstractConv2d_gradW...d=False}.0, 'W2': AbstractConv2d_gradW...d=False}.0, 'W4': AbstractConv2d_gradW...d=False}.0, 'W6': dot.0, 'W8': dot.0, 'b1': InplaceDimShuffle{1}.0, 'b3': InplaceDimShuffle{1}.0, 'b5': InplaceDimShuffle{1}.0, 'b7': InplaceDimShuffle{1}.0, 'b9': InplaceDimShuffle{1}.0}
    grad_fn = theano.function([input_var, target_var, target_var_B], grads_dict)
    # 正好是删去的最后两层的网络结构grads_B:[dot.0, InplaceDimShuffle{1}.0]
    grads_B_fn = theano.function([input_var, target_var, target_var_B], grads_B)

    test_acc = T.sum(T.eq(T.argmax(prediction_B, axis=1), target_var_B), dtype=theano.config.floatX)
    # val_fn 为攻击者私有的验证属性分类的正确性模型
    val_fn = theano.function([input_var, target_var_B], [loss_B, test_acc])

    return network, params, grad_fn, params_B, grads_B_fn, val_fn


def build_worker(input_shape, classes=2, lr=None, seed=54321):
    lasagne.random.set_rng(np.random.RandomState(seed))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network_dict = build_cnn(input_var, classes=classes, input_shape=input_shape)
    network = network_dict['fc2']
    prediction = lasagne.layers.get_output(network)
    prediction = lasagne.nonlinearities.softmax(prediction)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # save init
    params = lasagne.layers.get_all_params(network, trainable=True)

    if lr is not None:
        updates = lasagne.updates.sgd(loss, params, lr)
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        return network, params, train_fn

    grads = T.grad(loss, params)

    p_idx = 0
    grads_dict = dict()
    for p, g in zip(params, grads):
        key = p.name + str(p_idx)
        p_idx += 1
        grads_dict[key] = g
    #grads_dict: {'W0': (dmean/dW), 'W2': (dmean/dW), 'W4': (dmean/dW), 'W6': (dmean/dW), 'W8': (dmean/dW), 'b1': (dmean/db), 'b3': (dmean/db), 'b5': (dmean/db), 'b7': (dmean/db), 'b9': (dmean/db)}
    grad_fn = theano.function([input_var, target_var], grads_dict)
    # params [W, b, W, b, W, b, W, b, W, b]
    return network, params, grad_fn


def inf_data(x, y, batchsize, shuffle=False, y_b=None):
    while True:
        for b in iterate_minibatches(x, y, batchsize=batchsize, shuffle=shuffle, targets_B=y_b):
            yield b


def mix_inf_data(p_inputs, p_targets, np_inputs, np_targets, batchsize, mix_p=0.5):
    p_batchsize = int(mix_p * batchsize)
    np_batchsize = batchsize - p_batchsize

    print ('Mixing {} prop data with {} non prop data'.format(p_batchsize, np_batchsize))
    #根据mix_p的大小，决定一批次32个数据，多少个有属性，多少个没有属性，然后链接到一起
    p_gen = inf_data(p_inputs, p_targets, p_batchsize, shuffle=True)
    np_gen = inf_data(np_inputs, np_targets, np_batchsize, shuffle=True)

    while True:
        px, py = p_gen.__next__()
        npx, npy = np_gen.__next__()
        x = np.vstack([px, npx])
        y = np.concatenate([py, npy])
        yield x, y


def set_local(global_params, local_params_list):
    for params in local_params_list:
        for p, gp in zip(params, global_params):
            p.set_value(gp.get_value())

# add by wzx, partial download gradients from parameter server 
def set_local_partial(old, new, local_params_list, id, partial):
    # find 10% maxchanged and change old_global_params, use it to chan
    # change old to old + Dmax10(old,new)
    first = True
    for i,v in enumerate(old):
        if first == True:
            no = np.array(old[i]).flatten()
            nn = np.array(new[i]).flatten()
            first = False
        else:
            no = np.concatenate((no,np.array(old[i]).flatten()))
            nn = np.concatenate((nn,np.array(new[i]).flatten()))
    diff = abs(nn - no)
    max_index = np.argsort(-diff)
    needed = len(diff) * partial
    for i in range(len(diff)):
        if i < needed:
            # four dimesinal m,n,p,q
            if max_index[i] <= 863:
                m = math.floor(max_index[i] / 27)
                m_left = max_index[i] % 27
                n = math.floor(m_left / 9)
                n_left = m_left % 9
                p = math.floor(n_left / 3)
                q = n_left % 3
                old[0][m][n][p][q] = new[0][m][n][p][q]
            elif max_index[i] >= 864 and max_index[i] <= 895:
                pos = max_index[i] - 864
                old[1][pos] = new[1][pos]
            elif max_index[i] >= 896 and max_index[i] <= 19327:
                tmp = max_index[i] - 896
                m = math.floor(tmp / 288) 
                m_left = tmp % 288
                n = math.floor(m_left / 9)
                n_left = m_left % 9
                p = math.floor(n_left / 3)
                q = n_left % 3
                old[2][m][n][p][q] = new[2][m][n][p][q]
            elif max_index[i] >= 19328 and max_index[i] <= 19391:
                pos = max_index[i] - 19328
                old[3][pos] = new[3][pos]
            elif max_index[i] >= 19392 and max_index[i] <= 93119:
                tmp = max_index[i] - 19392
                m = math.floor(tmp / 576) 
                m_left = tmp % 576
                n = math.floor(m_left / 9)
                n_left = m_left % 9
                p = math.floor(n_left / 3)
                q = n_left % 3
                old[4][m][n][p][q] = new[4][m][n][p][q]
            elif max_index[i] >= 93120 and max_index[i] <= 93247:
                pos = max_index[i] - 93120
                old[5][pos] = new[5][pos]
            elif max_index[i] >= 93248 and max_index[i] <= 879679:
                tmp = max_index[i] - 93248
                m = math.floor(tmp / 256)
                n = tmp % 256
                old[6][m][n] = new[6][m][n]
            elif max_index[i] >= 879680 and max_index[i] <= 879935:
                pos = max_index[i] - 879680
                old[7][pos] = new[7][pos]
            elif max_index[i] >= 879936 and max_index[i] <= 880447:
                tmp = max_index[i] - 879936
                m = math.floor(tmp / 2)
                n = tmp % 2
                old[8][m][n] = new[8][m][n]
            elif max_index[i] >= 880448 and max_index[i] <= 880449:
                pos = max_index[i] - 880448
                old[9][pos] = new[9][pos]
        else:
            break

    for i, params in enumerate(local_params_list):
        if i == id:
            for p, g in zip(params, old):
                p_val = p.get_value()
                g = np.asarray(g)
                p.set_value(g)


def update_global(global_params, grads, lr):
    # upload all
    for p, g in zip(global_params, grads):
        p_val = p.get_value()
        g = np.asarray(g)
        p.set_value(p_val - g * np.float32(lr))

def update_global_wzx(old_global_params, global_params, grads, lr):
    # set old to new, change new to newer
    for p, g in zip(old_global_params, global_params):
            p_val = p.get_value()
            g_val = g.get_value()
            g_val = np.asarray(g_val)
            p.set_value(g_val)
    for p, g in zip(global_params, grads):
        p_val = p.get_value()
        g = np.asarray(g)
        p.set_value(p_val - g * np.float32(lr))


def add_nonprop(test_prop_indices, nonprop_indices, p_prop=0.7):
    n = len(test_prop_indices)
    n_to_add = int(n / p_prop) - n

    sampled_non_prop = np.random.choice(nonprop_indices, n_to_add, replace=False)
    nonprop_indices = np.setdiff1d(nonprop_indices, sampled_non_prop)
    return sampled_non_prop, nonprop_indices


def gradient_getter(data, p_g, p_indices, fn, batch_size=32, shuffle=True):
    X, y = data
    p_x, p_y = X[p_indices], y[p_indices]

    for batch in iterate_minibatches(p_x, p_y, batch_size, shuffle=shuffle):
        xx, yy = batch
        gs = fn(xx, yy)
        p_g.append(np.asarray(gs).flatten())


def gradient_getter_with_gen(data_gen, p_g, fn, iters=10, param_names=None):
    for _ in range(iters):
        xx, yy = next(data_gen)
        gs = fn(xx, yy)
        if isinstance(gs, dict):
            gs = collect_grads(gs, param_names)
        else:
            gs = np.asarray(gs).flatten()
        p_g.append(gs)


def gradient_getter_with_gen_multi(data_gen1, data_gen2, p_g, fn, iters=10, n_workers=5, param_names=None):
    for _ in range(iters):
        xx, yy = next(data_gen1)
        pgs = fn(xx, yy)

        if isinstance(pgs, dict):
            for key in pgs:
                pgs[key] = np.asarray(pgs[key])
        else:
            pgs = np.asarray(pgs).flatten()

        for _ in range(n_workers - 2):
            xx, yy = next(data_gen2)
            npgs = fn(xx, yy)
            if isinstance(npgs, dict):
                for key in npgs:
                    pgs[key] += np.asarray(npgs[key])
            else:
                npgs = np.asarray(npgs).flatten()
                pgs += npgs

        if isinstance(pgs, dict):
            pgs = collect_grads(pgs, param_names)

        p_g.append(pgs)


def collect_grads(grads_dict, param_names, avg_pool=False, pool_thresh=5000):
    g = []
    # first round: param_name: 'W0',shape:(32,3,3,3)
    for param_name in param_names:
        grad = grads_dict[param_name]
        grad = np.asarray(grad)
        # shape : (32, 3, 3, 3)
        shape = grad.shape

        if len(shape) == 1:
            continue

        grad = np.abs(grad)
        if len(shape) == 4:
            if shape[0] * shape[1] > pool_thresh:
                continue
            grad = grad.reshape(shape[0], shape[1], -1)

        if len(shape) > 2 or shape[0] * shape[1] > pool_thresh:
            if avg_pool:
                grad = np.mean(grad, -1)
            else:
                grad = np.max(grad, -1)

        g.append(grad.flatten())
    # 将g变成一维数组
    g = np.concatenate(g)
    return g


def aggregate_dicts(dicts, param_names):
    aggr_dict = dicts[0]
    # 将字典中的列表转换为np array
    for key in aggr_dict:
        aggr_dict[key] = np.asarray(aggr_dict[key])

    for d in dicts[1:]:
        for key in aggr_dict:
            aggr_dict[key] += np.asarray(d[key])

    return collect_grads(aggr_dict, param_names)


def train_multi_task_ps(data, num_iteration=6000, train_size=0.3, victim_id=0, seed=12345, warm_up_iters=100,
                        input_shape=(None, 3, 50, 50), n_workers=2, lr=0.01, attacker_id=1, filename="data",
                        p_prop=0.5, alpha_B=0., victim_all_nonprop=True, k=5):
    #splitted_X[0]包含的是victim的数据，splitted_X[1]包含的是attacker的数据
    #splitted_X[0][0]带有prop数据，splitted_X[0][1]不带有prop数据
    splitted_X, splitted_y, X_test, y_test = prepare_data_biased(data, train_size, n_workers, seed=seed,
                                                                 victim_all_nonprop=victim_all_nonprop,
                                                                 p_prop=p_prop)
    p_test = y_test[:, 1]
    y_test = y_test[:, 0]

    classes = len(np.unique(y_test))
    # build test network
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network_dict = build_cnn(input_var, classes=classes, input_shape=input_shape)
    network = network_dict['fc2']
    prediction = lasagne.layers.get_output(network)
    prediction = lasagne.nonlinearities.softmax(prediction)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # global_params 长度为10的列表, 包含所有参数
    global_params = lasagne.layers.get_all_params(network, trainable=True)
    global_params_values = lasagne.layers.get_all_param_values(network, trainable=True)
    global_grads = T.grad(loss, global_params)

    # save old global_param for use
    old_global_params = lasagne.layers.get_all_params(network, trainable=True)
    p_idx = 0
    grads_dict = dict()
    params_names = []
    for p, g in zip(global_params, global_grads):
        key = p.name + str(p_idx)
        params_names.append(key)
        p_idx += 1
        grads_dict[key] = g
    ps_dict = grads_dict
    # grads_dict:{'W0': (dmean/dW), 'W2': (dmean/dW), 'W4': (dmean/dW), 'W6': (dmean/dW), 'W8': (dmean/dW), 'b1': (dmean/db), 'b3': (dmean/db), 'b5': (dmean/db), 'b7': (dmean/db), 'b9': (dmean/db)}
    global_grad_fn = theano.function([input_var, target_var], grads_dict)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # build local workers
    worker_params = []
    worker_grad_fns = []
    data_gens = []

    for i in range(n_workers):
        print("epoch : " + str(i))
        if i == attacker_id:
            split_y = splitted_y[i]
            # net_attacker, p, f, b_params, b_grad_fn, pval_fn = build_worker_attacker(input_shape, classes=classes, alph_B=alpha_B,
            #                                                            infer_classes=len(np.unique(split_y[:, 1])), lr = 0.1)
            net_attacker, p, f = build_worker_attacker(input_shape, classes=classes, alph_B=alpha_B,
                                                            infer_classes=len(np.unique(split_y[:, 1])), lr = 0.1)
            data_gen = inf_data(splitted_X[i], split_y[:, 0], y_b=split_y[:, 1], batchsize=32, shuffle=True)
            print ('Participant {} with {} data'.format(i, len(splitted_X[i])))
            data_gens.append(data_gen)
        elif i == victim_id:
            net_victim, p, f = build_worker(input_shape, classes=classes, lr = 0.1)
            # vic_Xshape:(560, 3, 62, 47)
            vic_X = np.vstack([splitted_X[i][0], splitted_X[i][1]])
            vic_y = np.concatenate([splitted_y[i][0][:, 0], splitted_y[i][1][:, 0]])
            vic_p = np.concatenate([splitted_y[i][0][:, 1], splitted_y[i][1][:, 1]])
            data_gen = inf_data(vic_X, vic_y, y_b=vic_p, batchsize=32, shuffle=True)

            data_gen_p = inf_data(splitted_X[i][0], splitted_y[i][0][:, 0], batchsize=32, shuffle=True)
            data_gen_np = inf_data(splitted_X[i][1], splitted_y[i][1][:, 0], batchsize=32, shuffle=True)

            data_gens.append(data_gen)
            print ('Participant {} with {} data'.format(i, len(splitted_X[i][0]) + len(splitted_X[i][1])))
        else:
            p, f = build_worker(input_shape, classes=classes)
            data_gen = inf_data(splitted_X[i], splitted_y[i][:, 0], batchsize=32, shuffle=True)
            print ('Participant {} with {} data'.format(i, len(splitted_X[i])))
            data_gens.append(data_gen)

        worker_params.append(p)
        worker_grad_fns.append(f)
    # worker_params：[[W, b, W, b, W, b, W, b, W, ...], [W, b, W, b, W, b, W, b, W, ...]]结构相同，主任务可以正常训练
    #global_params ：[W, b, W, b, W, b, W, b, W, b]
    set_local(global_params, worker_params) # 一开始将所有参与者的参数都设置和参数服务器相同

    train_pg, train_npg = [], []
    test_pg, test_npg = [], []
    # 这里将全部11644个参数重新赋值给X,y
    X, y, _ = data

    # attacker's aux data
    X_adv, y_adv = splitted_X[attacker_id], splitted_y[attacker_id]
    # y_adv :shape:(7590, 2)。将第0列分为gender分给y_adv，第一列分为race分给p_adv
    p_adv = y_adv[:, 1]
    y_adv = y_adv[:, 0]

    indices = np.arange(len(X_adv))
    # 提取出attacker的119的prop data
    prop_indices = indices[p_adv == 1]
    # 提取出attacker 的7741个no prop data
    nonprop_indices = indices[p_adv == 0]
    # 定义生成器mix_p=0.2，也就是 一会日志出现 Mixing 6 prop data with 26 non prop data
    adv_gen = mix_inf_data(X_adv[prop_indices], splitted_y[attacker_id][prop_indices],
                           X_adv[nonprop_indices], splitted_y[attacker_id][nonprop_indices], batchsize=32, mix_p=0.2)
    # 将测试数据拼接到attacker train数据集上 X_adv size 7590 + 3494 = 11084,label y, property p size 也为11084
    X_adv = np.vstack([X_adv, X_test])
    y_adv = np.concatenate([y_adv, y_test])
    p_adv = np.concatenate([p_adv, p_test])

    indices = np.arange(len(p_adv))
    # 599个black，280分给victim，119还在attacker，剩余160在test里面， 119+160 = 279.train_prop_indices size为279
    train_prop_indices = indices[p_adv == 1]
    train_prop_gen = inf_data(X_adv[train_prop_indices], y_adv[train_prop_indices], 32, shuffle=True)

    indices = np.arange(len(p_test))
    nonprop_indices = indices[p_test == 0]
    # n_nonprop 为 3494-160，为3334，也就是不带属性black的测试数据为3334个
    n_nonprop = len(nonprop_indices)

    print ('Attacker prop data {}, non prop data {}'.format(len(train_prop_indices), n_nonprop))
    train_nonprop_gen = inf_data(X_test[nonprop_indices], y_test[nonprop_indices], 32, shuffle=True)

    train_mix_gens = []
    for train_mix_p in [0.4, 0.6, 0.8]:
        train_mix_gen = mix_inf_data(X_adv[train_prop_indices], y_adv[train_prop_indices],
                                     X_test[nonprop_indices], y_test[nonprop_indices], batchsize=32, mix_p=train_mix_p)
        train_mix_gens.append(train_mix_gen)

    start_time = time.time()
    for it in range(num_iteration):
        aggr_grad = []
        # set_local(global_params, worker_params)
        for i in range(n_workers):
            grad_fn = worker_grad_fns[i]
            data_gen = data_gens[i]
            if it == 0 and i == 0:
                print("first round no change")
            else:
                set_local_partial(new_pre_pre, new_pre,worker_params,i, 0.1) ### ！！！！download 梯度操作, 修改每个参与者训练本次任务都先进行梯度更新，更不是一轮直接更新
            for i_i in range(2):
                print("------------------------------")
                for t_i in range(10):
                    print(worker_params[i_i][1].get_value()[t_i])
            if i == attacker_id:
                batch = next(adv_gen)
                inputs, targets = batch
                targetsB = targets[:, 1]
                targets = targets[:, 0]
                grads_dict = grad_fn(inputs, targets, targetsB)
            elif i == victim_id:
                if it % k == 0:
                    inputs, targets = next(data_gen_p)# 第一轮迭代的是有属性的280个victim data
                else:
                    inputs, targets = next(data_gen_np)
                grads_dict = grad_fn(inputs, targets)
            else:
                inputs, targets = next(data_gen)
                grads_dict = grad_fn(inputs, targets)

            if i != attacker_id:
                # 梯度列表，列表中每一项都是当前轮一批次的某个参与者的梯度变化
                aggr_grad.append(grads_dict)

            #grads = [grads_dict[name] for name in params_names]
            #print("test here")
            # ttt = np.asarray(grads[1])
            #update_global(global_params, grads, lr) ### ！！！！upload 梯度操作，每一轮单个参与者进行训练后，进行部分梯度上传
            #print("test here2")
            # 获取网络参数失效lasagne.layers.get_all_param_values只能获取默认参数，所有的参数变化都在grads_dict里面
            new_pre_pre = global_params_values
            if i == victim_id:
                global_params_values = lasagne.layers.get_all_param_values(net_victim, trainable=True)
            else:
                global_params_values = lasagne.layers.get_all_param_values(net_attacker, trainable=True)
            new_pre = global_params_values
        

        if it >= warm_up_iters:
            # param_names ：['W0', 'b1', 'W2', 'b3', 'W4', 'b5', 'W6', 'b7', 'W8', 'b9']
            # test_gs shape: (5728,)
            test_gs = aggregate_dicts(aggr_grad, param_names=params_names)
            if it % k == 0:
                # 此时test_pg.append此test_gs
                test_pg.append(test_gs)
            else:
                test_npg.append(test_gs)

            if n_workers > 2:
                for train_mix_gen in train_mix_gens:
                    gradient_getter_with_gen_multi(train_mix_gen, train_nonprop_gen, train_pg, global_grad_fn,
                                                   iters=2, n_workers=n_workers, param_names=params_names)
                gradient_getter_with_gen_multi(train_prop_gen, train_nonprop_gen, train_pg, global_grad_fn,
                                               iters=2, n_workers=n_workers, param_names=params_names)
                gradient_getter_with_gen_multi(train_nonprop_gen, train_nonprop_gen, train_npg, global_grad_fn,
                                               iters=8, n_workers=n_workers, param_names=params_names)
            else:
                gradient_getter_with_gen(train_prop_gen, train_pg, global_grad_fn, iters=2,
                                         param_names=params_names) # 带有属性的数据集训练两次，截取部分梯度保存在 train_pg里面
                for train_mix_gen in train_mix_gens:
                    gradient_getter_with_gen(train_mix_gen, train_pg, global_grad_fn, iters=2,
                                             param_names=params_names) #注意到iters=2，所以每次执行此函数train_pg新增两行梯度变化数据，现在总共执行4次，故有8行数据。

                gradient_getter_with_gen(train_nonprop_gen, train_npg, global_grad_fn, iters=8,
                                         param_names=params_names) # train_npg的行也为8，因为iters=8,他们的列都有5728，是部分梯度截取后的结果
        # 当it满足条件时候，就开始进行模型的验证
        #if (it + 1) % 500 == 0 and it > 0:
        if it >= 0:
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0

            val_perr = 0
            val_pacc = 0
            val_pbatches = 0

            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            #for batch in iterate_minibatches(X_test, p_test, 500, shuffle=False):
            #    inputs, targets = batch
            #    err, acc = pval_fn(inputs, targets)
            #    val_perr += err
            #    val_pacc += acc
            #    val_pbatches += 1

            sys.stderr.write("Iteration {} of {} took {:.3f}s\n".format(it + 1, num_iteration,
                                                                        time.time() - start_time))
            sys.stderr.write("  test accuracy:\t\t{:.2f} %\n".format(val_acc / val_batches / 500 * 100))
            #sys.stderr.write("  p-test accuracy:\t\t{:.2f} %\n".format(val_pacc / val_pbatches / 500 * 100))
            #start_time = time.time()

    np.savez(SAVE_DIR + "{}.npz".format(filename),
             train_pg=train_pg, train_npg=train_npg, test_pg=test_pg, test_npg=test_npg)


if __name__ == '__main__':
    train_lfw()
