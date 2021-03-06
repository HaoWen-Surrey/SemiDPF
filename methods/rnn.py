import tensorflow as tf
import sonnet as snt

from utils.data_utils import *
from utils.method_utils import compute_sq_distance
import time

class RNN():
    def __init__(self, init_with_true_state=False, model='2lstm', **unused_kwargs):

        self.placeholders = {'o': tf.placeholder('float32', [None, None, 24, 24, 3], 'observations'),
                     'a': tf.placeholder('float32', [None, None, 3], 'actions'),
                     's': tf.placeholder('float32', [None, None, 3], 'states'),
                     'm': tf.placeholder('int32', [None, None], 'mask'),
                     'keep_prob': tf.placeholder('float32')}
        self.pred_states = None
        self.init_with_true_state = init_with_true_state
        self.model = model

        # build models
        # <-- observation
        self.encoder = snt.Sequential([
            snt.nets.ConvNet2D([16, 32, 64], [[3, 3]], [2], [snt.SAME], activate_final=True, name='encoder/convnet'),
            snt.BatchFlatten(),
            lambda x: tf.nn.dropout(x, self.placeholders['keep_prob']),
            snt.Linear(128, name='encoder/Linear'),
            tf.nn.relu,
        ])

        # <-- action
        if self.model == '2lstm':
            self.rnn1 = snt.LSTM(512)
            self.rnn2 = snt.LSTM(512)
        if self.model == '2gru':
            self.rnn1 = snt.GRU(512)
            self.rnn2 = snt.GRU(512)
        elif self.model == 'ff':
            self.ff_lstm_replacement = snt.Sequential([
                snt.Linear(512),
                tf.nn.relu,
                snt.Linear(512),
                tf.nn.relu])

        self.belief_decoder = snt.Sequential([
            snt.Linear(256),
            tf.nn.relu,
            snt.Linear(256),
            tf.nn.relu,
            snt.Linear(3)
        ])


    def fit(self, sess, data, model_path, split_ratio, seq_len, batch_size, num_epochs, patience, learning_rate, dropout_keep_ratio,labeledRatio, fold, **unused_kwargs):
        with open("result_dpf.txt", 'w') as file0:
            print("batch_size={}, seq_len={}, num_epochs={}, split_ratio={}, labeledRatio={}, fold={}".format(batch_size, seq_len,num_epochs,split_ratio, labeledRatio, fold), file=file0)

        # global mask
        shape_s = data['s'].shape
        # nuber of 0 and 1
        N1 = int(shape_s[0] * shape_s[1] * labeledRatio)
        N0 = shape_s[0] * shape_s[1] - N1
        arr = np.array([0] * N0 + [1] * N1)
        np.random.shuffle(arr)
        mask = arr.reshape(shape_s[0], shape_s[1])
        data['m'] = mask
        # preprocess data
        data = split_data(data, ratio=split_ratio)

        traindatalen = data['train']['s'].shape[0]
        valdatalen = data['val']['s'].shape[0]

        epoch_lengths = {'train': (traindatalen // batch_size) * fold, 'val': (valdatalen // batch_size)}
        batch_iterators = {'train': make_batch_iterator(data['train'], batch_size=batch_size),
                           'val': make_repeating_batch_iterator(data['val'])
                           }
        means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data['train'])

        self.connect_modules(means, stds, state_mins, state_maxs, state_step_sizes)

        # training

        sq_dist = compute_sq_distance(self.pred_states, self.placeholders['s'], state_step_sizes)
        maskp = tf.cast(self.placeholders['m'], tf.float32)
        losses = {'mse': tf.reduce_mean(maskp*sq_dist)/labeledRatio,
                  'mse_last': tf.reduce_mean(tf.sqrt(sq_dist[:, -1]))}

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(losses['mse'])
        # clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        train_op = optimizer.apply_gradients(gradients)

        init = tf.global_variables_initializer()
        sess.run(init)

        # save statistics and prepare saving variables
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        np.savez(os.path.join(model_path, 'statistics'), means=means, stds=stds, state_step_sizes=state_step_sizes,
                 state_mins=state_mins, state_maxs=state_maxs)
        saver = tf.train.Saver()
        save_path = model_path + '/best_validation'

        loss_keys = ['mse_last', 'mse']
        if split_ratio < 1.0:
            data_keys = ['train', 'val']
        else:
            data_keys = ['train']

        log = {dk: {lk: {'mean': [], 'se': []} for lk in loss_keys} for dk in data_keys}
        # time/iteration
        log_time = []

        best_val_loss = np.inf
        best_epoch = 0
        i = 0
        while i < num_epochs and i - best_epoch < patience:
            # training
            loss_lists = dict()
            store_batch=[]
            store_pred=[]
            starttime = time.time()
            for dk in data_keys:
                # don't train in the first epoch, just evaluate the initial parameters
                if dk == 'train' and i == 0:
                    continue
                loss_lists = {lk: [] for lk in loss_keys}
                for e in range(epoch_lengths[dk]):
                    batch = next(batch_iterators[dk])
                    if dk == 'train':
                        s_losses, _ , s_pred= sess.run([losses, train_op, self.pred_states], {**{self.placeholders[key]: batch[key] for key in 'osam'},
                                                                **{self.placeholders['keep_prob']: dropout_keep_ratio}})
                    else:
                        s_losses = sess.run(losses, {**{self.placeholders[key]: batch[key] for key in 'osam'},
                                                            **{self.placeholders['keep_prob']: 1.0}})
                    for lk in loss_keys:
                        loss_lists[lk].append(s_losses[lk])
                # after each epoch, compute and log statistics
                for lk in loss_keys:
                    log[dk][lk]['mean'].append(np.mean(loss_lists[lk]))
                    log[dk][lk]['se'].append(np.std(loss_lists[lk], ddof=1) / np.sqrt(epoch_lengths[dk]))

            endtime = time.time()
            log_time.append((endtime - starttime))
            # check whether the current model is better than all previous models
            if 'val' in data_keys:
                if log['val']['mse_last']['mean'][-1] < best_val_loss:
                    best_val_loss = log['val']['mse_last']['mean'][-1]
                    best_epoch = i
                    # save current model
                    saver.save(sess, save_path)
                    txt = 'epoch {:>3} >> '.format(i)
                else:
                    txt = 'epoch {:>3} == '.format(i)
            else:
                best_epoch = i
                saver.save(sess, save_path)
                txt = 'epoch {:>3} >> '.format(i)

            # after going through all data sets, do a print out of the current result
            for lk in loss_keys:
                txt += '{}: '.format(lk)
                for dk in data_keys:
                    if len(log[dk][lk]['mean']) > 0:
                        txt += '{:.2f}+-{:.2f}/'.format(log[dk][lk]['mean'][-1], log[dk][lk]['se'][-1])
                txt = txt[:-1] + ' -- '
            txt = txt + f'time: {(endtime - starttime):.03f}s'
            print(txt)
            with open("result_dpf.txt", 'a') as file0:
                print(txt,file=file0)

            np.savez(os.path.join(model_path, 'rmse_last'), rmse_last=log)
            i += 1
        np.savez(os.path.join(model_path, 'exe_time'), log_time=log_time)
        saver.restore(sess, save_path)

        return log


    def connect_modules(self, means, stds, state_mins, state_maxs, state_step_sizes):

        tracking_info = tf.concat([((self.placeholders['s'] - means['s']) / stds['s'])[:, :1, :], tf.zeros_like(self.placeholders['s'][:,1:,:])], axis=1)
        flag = tf.concat([tf.ones_like(self.placeholders['s'][:,:1,:1]), tf.zeros_like(self.placeholders['s'][:,1:,:1])], axis=1)

        preproc_o = snt.BatchApply(self.encoder)((self.placeholders['o'] - means['o']) / stds['o'])
        # include tracking info
        if self.init_with_true_state:
            preproc_o = tf.concat([preproc_o, tracking_info, flag], axis=2)

        preproc_a = snt.BatchApply(snt.BatchFlatten())(self.placeholders['a'] / stds['a'])
        preproc_ao = tf.concat([preproc_o, preproc_a], axis=-1)

        if self.model == '2lstm' or self.model == '2gru':
            lstm1_out, lstm1_final_state = tf.nn.dynamic_rnn(self.rnn1, preproc_ao, dtype=tf.float32)
            lstm2_out, lstm2_final_state = tf.nn.dynamic_rnn(self.rnn2, lstm1_out, dtype=tf.float32)
            belief_list = lstm2_out

        elif self.model == 'ff':
            belief_list = snt.BatchApply(self.ff_lstm_replacement)(preproc_ao)

        self.pred_states = snt.BatchApply(self.belief_decoder)(belief_list)
        self.pred_states = self.pred_states * stds['s'] + means['s']


    def predict(self, sess, batch, **unused_kwargs):
        return sess.run(self.pred_states, {**{self.placeholders[key]: batch[key] for key in 'osa'},
                                           **{self.placeholders['keep_prob']: 1.0}})

    def load(self, sess, model_path, model_file='best_validation', statistics_file='statistics.npz', connect_and_initialize=True):

        # build the tensorflow graph
        if connect_and_initialize:
            # load training data statistics (which are needed to build the tf graph)
            statistics = dict(np.load(os.path.join(model_path, statistics_file),allow_pickle=True))
            for key in statistics.keys():
                if statistics[key].shape == ():
                    statistics[key] = statistics[key].item()  # convert 0d array of dictionary back to a normal dictionary

            # connect all modules into the particle filter
            self.connect_modules(**statistics)
            init = tf.global_variables_initializer()
            sess.run(init)

        # load variables
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in all_vars:
            print("%s %r %s" % (v, v, v.shape))

        # restore variable values
        saver = tf.train.Saver()  # <- var list goes in here
        saver.restore(sess, os.path.join(model_path, model_file))
