import tensorflow as tf

from methods.dpf import DPF
from methods.rnn import RNN
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state, split_data, make_repeating_batch_iterator
from utils.exp_utils import get_default_hyperparams
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

def train_dpf(task='nav01', data_path='../data/100s', model_path='../models/tmp', plot=False):
    with open("result_dpf.txt", 'w') as file0:
        print(f'task={task}', file=file0)

    # load training data and add noise
    train_data = load_data(data_path=data_path, filename=task + '_train')
    noisy_train_data = noisyfy_data(train_data)

    # reset tensorflow graph
    tf.reset_default_graph()
    tf.set_random_seed(100)
    np.random.seed(200)
    random.seed(300)

    # instantiate method
    hyperparams = get_default_hyperparams()
    method = DPF(**hyperparams['global'])

    with tf.Session() as session:
        # train method and save result in model_path
        method.fit(session, noisy_train_data, model_path, **hyperparams['train'], plot_task=task, plot=plot)


def test_dpf(task='nav01', data_path='../data/100s', model_path='../models/tmp'):

    # load test data
    test_data = load_data(data_path=data_path, filename=task + '_test')
    noisy_test_data = noisyfy_data(test_data)
    test_batch_iterator = make_repeating_batch_iterator(noisy_test_data)

    # reset tensorflow graph
    tf.reset_default_graph()
    tf.set_random_seed(100)
    np.random.seed(200)
    random.seed(300)

    # instantiate method
    hyperparams = get_default_hyperparams()
    method = DPF(**hyperparams['global'])

    with tf.Session() as session:
        # load method and apply to new data
        method.load(session, model_path)
        for i in range(31):
            test_batch = next(test_batch_iterator)
            pred, s_particle_list, s_particle_probs_list = method.predict(session, test_batch, **hyperparams['test'],return_particles=True)
            np.savez("semi_test" + str(i) + ".npz", weight=s_particle_probs_list, particle=s_particle_list, pred=pred, batch=test_batch)

if __name__ == '__main__':
    train_dpf(plot=False)
    test_dpf()
    print("over")
