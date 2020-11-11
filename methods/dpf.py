import os
import numpy as np
import sonnet as snt
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import warnings
import time

warnings.filterwarnings("ignore")

from utils.data_utils import wrap_angle, compute_staticstics, split_data, make_batch_iterator, make_repeating_batch_iterator
from utils.method_utils import atan2, compute_sq_distance, et_distance
from utils.plotting_utils import plot_maze, show_pause

class DPF():

    def __init__(self, init_with_true_state):
        """
        :param init_with_true_state:
        :param learn_odom:
        :param use_proposer:
        :param propose_ratio:
        :param particle_std:
        :param proposer_keep_ratio:
        :param min_obs_likelihood:
        """

        # store hyperparameters which are needed later
        self.init_with_true_state = init_with_true_state

        # define some more parameters and placeholders
        self.state_dim = 3
        self.hidden_size =128
        self.placeholders = {'o': tf.placeholder('float32', [None, None, 24, 24, 3], 'observations'),
                             'a': tf.placeholder('float32', [None, None, 3], 'actions'),
                             's': tf.placeholder('float32', [None, None, 3], 'states'),
                             'm': tf.placeholder('int32', [None, None], 'mask'),
                             'epoch':tf.placeholder('float32'),
                             'num_particles': tf.placeholder('float32'),
                             'keep_prob': tf.placeholder_with_default(tf.constant(1.0), []),
                             }
        self.num_particles_float = self.placeholders['num_particles']
        self.num_particles = tf.to_int32(self.num_particles_float)
        # build learnable modules
        self.build_modules()

    def build_modules(self):
        """
        :param min_obs_likelihood:
        :param proposer_keep_ratio:
        :return: None
        """

        # conv net for encoding the image
        self.encoder = snt.Sequential([
            snt.nets.ConvNet2D([16, 32, 64], [[3, 3]], [2], [snt.SAME], activate_final=True,
                               name='encoder/convnet'),
            snt.BatchFlatten(),
            lambda x: tf.nn.dropout(x, self.placeholders['keep_prob']),
            snt.Linear(self.hidden_size, name='encoder/linear'),
            tf.nn.relu
        ])

        self.decoder = snt.Sequential([
            snt.Linear(3 * 3 * 64, name='decoder/linear'),
            tf.nn.relu,
            snt.BatchReshape(shape=(3, 3, 64), preserve_dims=1, name='decoder/reshape'),
            snt.nets.ConvNet2DTranspose([32, 16, 3], [6, 12, 24], [[3, 3]], [2], [snt.SAME], activate_final=True,
                                        name='decoder/convnet')
        ]
        )

        self.st_to_et = snt.nets.MLP([16, 32, 64, self.hidden_size], activate_final=True, name='st_to_et')

    def measurement_update(self, encoding, particles, means, stds):
        """
        Compute the likelihood of the encoded observation for each particle.

        :param encoding: encoding of the observation
        :param particles:
        :param means:
        :param stds:
        :return: observation likelihood
        """

        # prepare input (normalize particles poses and repeat encoding per particle)
        particle_input = self.transform_particles_as_input(particles, means, stds)
        e_t = snt.BatchApply(self.st_to_et)(particle_input)

        encoding_input = tf.tile(encoding[:, tf.newaxis, :], [1, tf.shape(particles)[1], 1])
        obs_likelihood = 1/(1e-8+et_distance(encoding_input, e_t))

        return obs_likelihood, encoding_input, e_t


    def transform_particles_as_input(self, particles, means, stds):
        return tf.concat([
                   (particles[:, :, :2] - means['s'][:, :, :2]) / stds['s'][:, :, :2],  # normalized pos
                   tf.cos(particles[:, :, 2:3]),  # cos
                   tf.sin(particles[:, :, 2:3])], # sin
                  axis=-1)

    def propose_particles(self, encoding, num_particles, state_mins, state_maxs):
        duplicated_encoding = tf.tile(encoding[:, tf.newaxis, :], [1, num_particles, 1])
        proposed_particles = snt.BatchApply(self.particle_proposer)(duplicated_encoding)
        proposed_particles = tf.concat([
            proposed_particles[:,:,:1] * (state_maxs[0] - state_mins[0]) / 2.0 + (state_maxs[0] + state_mins[0]) / 2.0,
            proposed_particles[:,:,1:2] * (state_maxs[1] - state_mins[1]) / 2.0 + (state_maxs[1] + state_mins[1]) / 2.0,
            atan2(proposed_particles[:,:,2:3], proposed_particles[:,:,3:4])], axis=2)
        return proposed_particles

    def motion_update(self, actions, particles, means, stds, state_step_sizes,std_x, std_y, std_t, stop_sampling_gradient=False):
        """
        Move particles according to odometry info in actions. Add learned noise.

        :param actions:
        :param particles:
        :param means:
        :param stds:
        :param state_step_sizes:
        :param stop_sampling_gradient:
        :return: moved particles
        """

        # 1. SAMPLE NOISY ACTIONS
        actions1=actions
        # add dimension for particles
        actions1 = actions1[:, tf.newaxis, :]
        noisy_actions = tf.tile(actions1, [1, tf.shape(particles)[1], 1])

        # 2. APPLY NOISY ACTIONS
        # compute sin and cos of the particles
        theta = particles[:, :, 2:3]
        sin_theta = tf.sin(theta)
        cos_theta = tf.cos(theta)

        new_x = particles[:, :, 0:1] + (
                    noisy_actions[:, :, 0:1] * cos_theta + noisy_actions[:, :, 1:2] * sin_theta) + tf.random_normal(
            tf.shape(particles[:, :, 0:1]), mean=0.0, stddev=std_x)
        new_y = particles[:, :, 1:2] + (
                    noisy_actions[:, :, 0:1] * sin_theta - noisy_actions[:, :, 1:2] * cos_theta) + tf.random_normal(
            tf.shape(particles[:, :, 1:2]), mean=0.0, stddev=std_y)
        new_theta = wrap_angle(particles[:, :, 2:3] + noisy_actions[:, :, 2:3]) + tf.random_normal(
            tf.shape(particles[:, :, 2:3]), mean=0.0, stddev=std_t)

        noise_x = new_x - (particles[:, :, 0:1] + (
                    noisy_actions[:, :, 0:1] * cos_theta + noisy_actions[:, :, 1:2] * sin_theta))
        noise_y = new_y - (particles[:, :, 1:2] + (
                    noisy_actions[:, :, 0:1] * sin_theta - noisy_actions[:, :, 1:2] * cos_theta))
        noise_t = new_theta - wrap_angle(particles[:, :, 2:3] + noisy_actions[:, :, 2:3])
        noise_s = tf.concat([noise_x, noise_y, noise_t], axis=-1)

        moved_particles = tf.concat([new_x, new_y, new_theta], axis=-1)

        return moved_particles, noise_s


    def compile_training_stages(self, sess, batch_iterators, particle_list, particle_probs_list,additional_probs_list, encodings, means, stds, state_step_sizes, state_mins, state_maxs, learning_rate, plot_task,seq_len, labeledRatio, block_len, num_particles,std_x, std_y, std_t, noises_list,aeloss, particle_path, noises_path,particle_probs_path, additional_probs_path):

        # TRAINING!
        losses = dict()
        train_stages = dict()

        # END-TO-END TRAINING
        # labeled state loss, mask shape: (32,20)
        mask = tf.cast(self.placeholders['m'], tf.float32)

        if 0.0==labeledRatio:
            # unsupervised learning
            # stage 2: update loss 4
            Q = self.compute_block_density(encodings, particle_list, particle_probs_list,additional_probs_list, means, stds,state_mins, state_maxs, seq_len,
                                           block_len, num_particles, std_x, std_y, std_t, noises_list)

            l4e2emle = -tf.reduce_mean(tf.reduce_sum(Q, axis=-1))
            losses['l4e2emle'] = l4e2emle

            if aeloss:
                losses['ae'] = self.compute_aeloss(particle_list, means, stds, seq_len)

            # second loss (which we will monitor during execution)
            pred = self.particles_to_state(particle_list, particle_probs_list)

            sq_distance1 = compute_sq_distance(pred[:, -1, :], self.placeholders['s'][:, -1, :], state_step_sizes)
            losses['mse_last'] = tf.reduce_mean(sq_distance1)

            optimizerl5 = tf.train.AdamOptimizer(learning_rate)
            if labeledRatio < 1.0:
                lamda1 = 0.01  # penalty term of loss4
                lamda2 = 200  # penalty term of lossae
                if aeloss:
                    losses['l5e2e'] = lamda1 * l4e2emle + lamda2 * losses['ae']
                else:
                    losses['l5e2e'] = lamda1 * l4e2emle

            # put everything together
            train_stages['train_unsup'] = {
                'train_op_l5': optimizerl5.minimize(losses['l5e2e']),
                'batch_iterator_names': {'train': 'train', 'val': 'val'},
                'monitor_losses': ['mse_last', 'l5e2e'],
                'validation_loss': ['mse_last'],
                'plot': lambda e: self.plot_particle_filter(sess, next(batch_iterators['val']), plot_task,e) if e % 1 == 0 else None
            }

        if 0.0<labeledRatio:

            # define losses and optimizer
            # first loss (which is being optimized)
            sq_distance = compute_sq_distance(particle_list, self.placeholders['s'][:, :, tf.newaxis, :], state_step_sizes)
            activations = particle_probs_list[:, :] / tf.sqrt(2 * np.pi * self.particle_std ** 2) * tf.exp(
                -sq_distance / (2.0 * self.particle_std ** 2))
            losses['mle'] = tf.reduce_mean(-mask *tf.log(1e-16 + tf.reduce_sum(activations, axis=2, name='loss')))/labeledRatio

            optimizer = tf.train.AdamOptimizer(learning_rate)

            # put everything together
            train_stages['train_e2e'] = {
                         'train_op': optimizer.minimize(losses['mle']),
                         'batch_iterator_names': {'train': 'train', 'val': 'val'},
                         'plot': lambda e: self.plot_particle_filter(sess, next(batch_iterators['val']), plot_task,e) if e % 1 == 0 else None
                         }

            # Semi End2End training
            l3e2e = losses['mle']
            losses['l3e2e']=l3e2e

            if labeledRatio < 1.0:
                # stage 2: update loss 4
                Q = self.compute_block_density(encodings, particle_list, means, stds, particle_probs_list,
                                               additional_probs_list, state_mins, state_maxs, seq_len, block_len, std_x,
                                               std_y, std_t, noises_list, particle_path, noises_path,
                                               particle_probs_path, additional_probs_path)

                l4e2emle=-tf.reduce_mean(tf.reduce_sum(Q,axis=-1))
                losses['l4e2emle']=l4e2emle

                if aeloss:
                    losses['ae']=self.compute_aeloss(particle_list,means,stds,seq_len)

            # second loss (which we will monitor during execution)
            pred = self.particles_to_state(particle_list, particle_probs_list)

            sq_distance1 = compute_sq_distance(pred[:, -1, :], self.placeholders['s'][:, -1, :], state_step_sizes)
            losses['mse_last'] = tf.reduce_mean(tf.sqrt(sq_distance1))

            optimizerl5 = tf.train.AdamOptimizer(learning_rate)
            if labeledRatio < 1.0:
                lamda1=0.01 # penalty term of loss4
                lamda2=200 # penalty term of lossae
                if aeloss:
                    losses['l5e2e'] = l3e2e*10 + lamda1 * l4e2emle+ lamda2* losses['ae']
                else:
                    losses['l5e2e'] =l3e2e + lamda1*l4e2emle
            else:
                losses['l5e2e'] =l3e2e

            # put everything together
            train_stages['train_semie2e'] = {
                'train_op_l5': optimizerl5.minimize(losses['l5e2e']),
                'batch_iterator_names': {'train': 'train', 'val': 'val'},
                'plot': lambda e: self.plot_particle_filter(sess, next(batch_iterators['val']),
                                                            plot_task,e) if e % 1 == 0 else None
            }

        return losses, train_stages

    def compute_aeloss(self, particle_list, means, stds, seq_len):
        orig_o=((self.placeholders['o'] - means['o']) / stds['o'])
        decoder_o = snt.BatchApply(self.decoder)(snt.BatchApply(self.encoder)(orig_o))
        o_aeloss=tf.reduce_mean((orig_o-decoder_o)**2)
        return o_aeloss

    def compute_block_density(self, encodings, particle_list, means, stds, particle_probs_list, additional_probs_list,
                              state_mins, state_maxs, seq_len, block_len, std_x, std_y, std_t, noises_list,
                              particle_path, noises_path, particle_probs_path, additional_probs_path):
        """
        Compute the density of the block.

        :param encoding: encoding of the observation
        :param particles:
        :param means:
        :param stds:
        :return: observation likelihood
        """
        # block index
        b = 0
        # log constant
        log_c = -tf.log(math.sqrt(2 * math.pi))
        # initial stationary distribution density
        mu_s = 1.0
        for d in range(self.state_dim):
            mu_s *= 1.0 / (state_maxs[d] - state_mins[d])
        # Q value
        Q = 0
        for k in range(1, seq_len):
            if (k + 1) % block_len == 0:
                if b == 0:
                    lki0 = tf.ones([self.batch_size, self.num_particles]) / self.num_particles_float
                    logyita = tf.log(mu_s * lki0)
                else:
                    lki = additional_probs_path[:, b * block_len, :]
                    # compute the prior density
                    logpriorx1 = log_c - tf.log(std_x) - (noises_path[:, b * block_len, :, 0]) ** 2 / (2 * (std_x) ** 2)
                    logpriory1 = log_c - tf.log(std_y) - (noises_path[:, b * block_len, :, 1]) ** 2 / (2 * (std_y) ** 2)
                    logpriort1 = log_c - tf.log(std_t) - (noises_path[:, b * block_len, :, 2]) ** 2 / (2 * (std_t) ** 2)
                    logprior1 = logpriorx1 + logpriory1 + logpriort1
                    logyita = logprior1 + tf.log(lki)
                m = k - block_len + 2
                for i in range(m, k + 1):
                    lki1 = additional_probs_path[:, i, :]
                    # compute the prior density
                    logpriorx = log_c - tf.log(std_x) - (noises_path[:, i, :, 0]) ** 2 / (2 * (std_x) ** 2)
                    logpriory = log_c - tf.log(std_y) - (noises_path[:, i, :, 1]) ** 2 / (2 * (std_y) ** 2)
                    logpriort = log_c - tf.log(std_t) - (noises_path[:, i, :, 2]) ** 2 / (2 * (std_t) ** 2)
                    logprior = logpriorx + logpriory + logpriort
                    logyita = logyita + (logprior + tf.log(lki1))
                Q = Q + particle_probs_path[:, k, :] * logyita
                b = b + 1
        return Q / b

    def compute_ESS(self, particle_probs_list):
        batch_size=particle_probs_list.shape[0]
        seq_len=particle_probs_list.shape[1]
        num_particles=particle_probs_list.shape[2]
        ESS=np.mean(1.0/np.sum(particle_probs_list**2,axis=-1),axis=0)
        return ESS

    def load(self, sess, model_path, model_file='best_validation', statistics_file='statistics.npz', connect_and_initialize=True, modules=('encoder', 'decoder', 'st_to_et')):

        if type(modules) not in [type(list()), type(tuple())]:
            raise Exception('modules must be a list or tuple, not a ' + str(type(modules)))

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
        else:
            statistics = None

        # load variables
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vars_to_load = []
        loaded_modules = set()
        for v in all_vars:
            for m in modules:
                if m in v.name:
                    vars_to_load.append(v)
                    loaded_modules.add(m)

        print('Loading these modules:', loaded_modules)

        print('%s %s' % (model_path, model_file))
        print('%r %r' % (model_path, model_file))

        # restore variable values
        saver = tf.train.Saver(vars_to_load)  # <- var list goes in here
        saver.restore(sess, os.path.join(model_path, model_file))

        print('Loaded the following variables:')
        for v in vars_to_load:
            print(v.name)

        return statistics

    def fit(self, sess, data, model_path, train_individually, train_e2e,train_semie2e,train_unsup, split_ratio, seq_len, batch_size,epoch_length, num_epochs, patience, learning_rate, dropout_keep_ratio, num_particles, particle_std, labeledRatio,fold, block_len,std_x, std_y, std_t,aeloss,plot_task=None, plot=False):

        with open("result_dpf.txt", 'a') as file0:
            print("train_e2e={}, train_semie2e={}, batch_size={}, seq_len={}, num_epochs={}, block_len={}, labeledRatio={}, split_ratio={}".format(train_e2e,train_semie2e,batch_size, seq_len, num_epochs, block_len, labeledRatio,split_ratio), file=file0)
        self.particle_std = particle_std

        global mask
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
        epoch_lengths = {'train': (traindatalen//batch_size)*fold, 'val': (valdatalen//batch_size)}
        batch_iterators = {'train': make_batch_iterator(data['train'], batch_size=batch_size),
                           'val': make_repeating_batch_iterator(data['val'])
                           }

        # compute some statistics of the training data
        means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data['train'])

        # build the tensorflow graph by connecting all modules in the particles filter
        particles, particle_probs, encodings, particle_list, particle_probs_list, additional_probs_list, etRange_list,noises_list,encoding_input_list,e_t_list, particle_path, noises_path,particle_probs_path, additional_probs_path = self.connect_modules(means, stds, state_mins, state_maxs, state_step_sizes,std_x, std_y, std_t)

        # define losses and train stages for different ways of training (e.g. training individual models and e2e training)
        losses, train_stages = self.compile_training_stages(sess, batch_iterators, particle_list,
                                                            particle_probs_list, additional_probs_list,
                                                            encodings, means, stds,
                                                            state_step_sizes, state_mins,
                                                            state_maxs, learning_rate, plot_task,
                                                            seq_len, labeledRatio, block_len, num_particles, std_x,
                                                            std_y, std_t, noises_list, aeloss, particle_path,
                                                            noises_path, particle_probs_path, additional_probs_path)

        # initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # save statistics and prepare saving variables
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        np.savez(os.path.join(model_path, 'statistics'), means=means, stds=stds, state_step_sizes=state_step_sizes,
                 state_mins=state_mins, state_maxs=state_maxs,std_x=std_x, std_y=std_y, std_t=std_t)
        saver = tf.train.Saver()
        save_path = os.path.join(model_path, 'best_validation')

        # define the training curriculum
        curriculum = []
        if train_individually:
            if self.learn_odom:
                curriculum += ['train_odom']
            curriculum += ['train_motion_sampling']
            curriculum += ['train_measurement_model']
            if self.use_proposer:
                curriculum += ['train_particle_proposer']
        if train_e2e:
            curriculum += ['train_e2e']
        if train_semie2e:
            curriculum += ['train_semie2e']
        if train_unsup:
            curriculum += ['train_unsup']

        # split data for early stopping
        data_keys = ['train']
        if split_ratio < 1.0:
            data_keys.append('val')

        stat_et=dict()
        stat_et['mean']=[]
        stat_et['std']=[]

        if labeledRatio < 1:
            lossk = ['mse_last','l5', 'l3', 'l4','lae']
        else:
            lossk = ['mse_last', 'l3']

        stat_l3l4={dk:{lk:{'mean':[], 'std':[]} for lk in lossk} for dk in data_keys}

        # time/iteration
        log_time = []

        # go through curriculum
        for c in curriculum:
            stage = train_stages[c]
            best_val_loss = np.inf
            best_epoch = 0
            epoch = 0

            while epoch < num_epochs:
            # while epoch < num_epochs and epoch - best_epoch < patience:
                starttime = time.time()
                # training
                for dk in data_keys:
                    # don't train in the first epoch, just evaluate the initial parameters
                    if dk == 'train' and epoch == 0:
                        continue
                    # set up loss lists which will be filled during the epoch
                    logl3l4 = {dk: {lk: [] for lk in lossk} for dk in data_keys}
                    #print the range of et and e_t
                    log_et = []
                    # # compute the Effective Sample Size
                    # log_ESS = []

                    for e in range(epoch_lengths[dk]):
                        # t0 = time.time()
                        # pick a batch from the right iterator
                        batch = next(batch_iterators[stage['batch_iterator_names'][dk]])

                        # define the inputs and train/run the model
                        input_dict = {**{self.placeholders[key]: batch[key] for key in 'osam'},
                                      **{self.placeholders['num_particles']: num_particles},
                                      }
                        input_dict[self.placeholders['epoch']] = epoch*1.0
                        if dk == 'train':
                            input_dict[self.placeholders['keep_prob']] = dropout_keep_ratio
                        #monitor_losses = {l: losses[l] for l in stage['monitor_losses']}
                        if dk == 'train':
                            if c == 'train_semie2e':
                                lmse_last, l3, _, etRange_s = sess.run(
                                    [losses['mse_last'], losses['l3e2e'], stage['train_op_l5'], etRange_list],
                                    input_dict)
                                # log of etRange
                                log_et.append(np.mean(1 - 1 / etRange_s))
                                # # ESS
                                # ESS=self.compute_ESS(particle_probs_list_s)
                                # log_ESS.append(ESS)
                                logl3l4['train']['mse_last'].append(lmse_last)
                                logl3l4['train']['l3'].append(l3)
                                if labeledRatio < 1.0:
                                    l4,l5 = sess.run([losses['l4e2emle'],losses['l5e2e']], input_dict)
                                    logl3l4['train']['l4'].append(l4)
                                    logl3l4['train']['l5'].append(l5)
                                    if aeloss:
                                        lae= sess.run(losses['ae'], input_dict)
                                        logl3l4['train']['lae'].append(lae)
                            elif c == 'train_e2e':
                                lmse_last, l3, _, etRange_s, particle_probs_list_s = sess.run([losses['mse_last'], losses['mle'], stage['train_op'], etRange_list,particle_probs_list], input_dict)
                                # log of etRange
                                log_et.append(np.mean(1 - 1 / etRange_s))
                                # # ESS
                                # ESS = self.compute_ESS(particle_probs_list_s)
                                # log_ESS.append(ESS)
                                logl3l4['train']['mse_last'].append(lmse_last)
                                logl3l4['train']['l3'].append(l3)

                            elif c == 'train_unsup':
                                lmse_last, _, etRange_s, particle_probs_list_s = sess.run(
                                    [losses['mse_last'], stage['train_op_l5'], etRange_list, particle_probs_list],
                                    input_dict)
                                # log of etRange
                                log_et.append(np.mean(1 - 1 / etRange_s))
                                # # ESS
                                # log_ESS.append(particle_probs_list_s)
                                logl3l4['train']['mse_last'].append(lmse_last)
                                if labeledRatio < 1:
                                    l4, l5 = sess.run([losses['l4e2emle'], losses['l5e2e']], input_dict)
                                    logl3l4['train']['l4'].append(l4)
                                    logl3l4['train']['l5'].append(l5)
                                    if aeloss:
                                        lae = sess.run(losses['ae'], input_dict)
                                        logl3l4['train']['lae'].append(lae)
                        else:
                            if c == 'train_semie2e':
                                if labeledRatio < 1:
                                    lmse_last,l5, l3, l4 = sess.run([losses['mse_last'], losses['l5e2e'], losses['l3e2e'], losses['l4e2emle']],
                                                                 input_dict)
                                    logl3l4['val']['mse_last'].append(lmse_last)
                                    logl3l4['val']['l3'].append(l3)
                                    logl3l4['val']['l4'].append(l4)
                                    logl3l4['val']['l5'].append(l5)
                                    if aeloss:
                                        lae= sess.run(losses['ae'], input_dict)
                                        logl3l4['val']['lae'].append(lae)
                                else:
                                    lmse_last, l3 = sess.run([losses['mse_last'], losses['l3e2e']], input_dict)
                                    logl3l4['val']['mse_last'].append(lmse_last)
                                    logl3l4['val']['l3'].append(l3)
                            elif c == 'train_e2e':
                                lmse_last, l3 = sess.run([losses['mse_last'], losses['mle']], input_dict)
                                logl3l4['val']['mse_last'].append(lmse_last)
                                logl3l4['val']['l3'].append(l3)
                            elif c == 'train_unsup':
                                lmse_last, l5, l4 = sess.run(
                                    [losses['mse_last'], losses['l5e2e'], losses['l4e2emle']],
                                    input_dict)
                                logl3l4['val']['mse_last'].append(lmse_last)
                                logl3l4['val']['l4'].append(l4)
                                logl3l4['val']['l5'].append(l5)
                                if aeloss:
                                    lae = sess.run(losses['ae'], input_dict)
                                    logl3l4['val']['lae'].append(lae)
                    if dk == 'train':
                        if c == 'train_unsup':
                            # after each epoch, compute the etRange statistics
                            stat_et['mean'].append(np.mean(log_et))
                            stat_et['std'].append(np.std(log_et))
                            stat_l3l4['train']['mse_last']['mean'].append(np.mean(logl3l4['train']['mse_last']))
                            stat_l3l4['train']['mse_last']['std'].append(np.std(logl3l4['train']['mse_last']))
                            stat_l3l4['train']['l4']['mean'].append(np.mean(logl3l4['train']['l4']))
                            stat_l3l4['train']['l4']['std'].append(np.std(logl3l4['train']['l4']))
                            stat_l3l4['train']['l5']['mean'].append(np.mean(logl3l4['train']['l5']))
                            stat_l3l4['train']['l5']['std'].append(np.std(logl3l4['train']['l5']))
                            if aeloss:
                                stat_l3l4['train']['lae']['mean'].append(np.mean(logl3l4['train']['lae']))
                                stat_l3l4['train']['lae']['std'].append(np.std(logl3l4['train']['lae']))
                        else:
                            # after ech epoch, compute the etRange statistics
                            stat_et['mean'].append(np.mean(log_et))
                            stat_et['std'].append(np.std(log_et))
                            stat_l3l4['train']['mse_last']['mean'].append(np.mean(logl3l4['train']['mse_last']))
                            stat_l3l4['train']['mse_last']['std'].append(np.std(logl3l4['train']['mse_last']))
                            stat_l3l4['train']['l3']['mean'].append(np.mean(logl3l4['train']['l3']))
                            stat_l3l4['train']['l3']['std'].append(np.std(logl3l4['train']['l3']))
                            if labeledRatio < 1 and c == 'train_semie2e':
                                stat_l3l4['train']['l4']['mean'].append(np.mean(logl3l4['train']['l4']))
                                stat_l3l4['train']['l4']['std'].append(np.std(logl3l4['train']['l4']))
                                stat_l3l4['train']['l5']['mean'].append(np.mean(logl3l4['train']['l5']))
                                stat_l3l4['train']['l5']['std'].append(np.std(logl3l4['train']['l5']))
                                if aeloss:
                                    stat_l3l4['train']['lae']['mean'].append(np.mean(logl3l4['train']['lae']))
                                    stat_l3l4['train']['lae']['std'].append(np.std(logl3l4['train']['lae']))
                    else:
                        if c == 'train_unsup':
                            stat_l3l4['val']['mse_last']['mean'].append(np.mean(logl3l4['val']['mse_last']))
                            stat_l3l4['val']['mse_last']['std'].append(np.std(logl3l4['val']['mse_last']))
                            stat_l3l4['val']['l4']['mean'].append(np.mean(logl3l4['val']['l4']))
                            stat_l3l4['val']['l4']['std'].append(np.std(logl3l4['val']['l4']))
                            stat_l3l4['val']['l5']['mean'].append(np.mean(logl3l4['val']['l5']))
                            stat_l3l4['val']['l5']['std'].append(np.std(logl3l4['val']['l5']))
                            if aeloss:
                                stat_l3l4['val']['lae']['mean'].append(np.mean(logl3l4['val']['lae']))
                                stat_l3l4['val']['lae']['std'].append(np.std(logl3l4['val']['lae']))
                        else:
                            stat_l3l4['val']['mse_last']['mean'].append(np.mean(logl3l4['val']['mse_last']))
                            stat_l3l4['val']['mse_last']['std'].append(np.std(logl3l4['val']['mse_last']))
                            stat_l3l4['val']['l3']['mean'].append(np.mean(logl3l4['val']['l3']))
                            stat_l3l4['val']['l3']['std'].append(np.std(logl3l4['val']['l3']))
                            if labeledRatio < 1 and c == 'train_semie2e':
                                stat_l3l4['val']['l4']['mean'].append(np.mean(logl3l4['val']['l4']))
                                stat_l3l4['val']['l4']['std'].append(np.std(logl3l4['val']['l4']))
                                stat_l3l4['val']['l5']['mean'].append(np.mean(logl3l4['val']['l5']))
                                stat_l3l4['val']['l5']['std'].append(np.std(logl3l4['val']['l5']))
                                if aeloss:
                                    stat_l3l4['val']['lae']['mean'].append(np.mean(logl3l4['val']['lae']))
                                    stat_l3l4['val']['lae']['std'].append(np.std(logl3l4['val']['lae']))
                endtime = time.time()
                log_time.append((endtime - starttime))
                # check whether the current model is better than all previous models
                if 'val' in data_keys:
                    current_val_loss =stat_l3l4['val']['mse_last']['mean'][-1]
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        best_epoch = epoch
                        # save current model
                        saver.save(sess, save_path)
                        txt = 'epoch {:>3} >> '.format(epoch)
                        # stage['plot'](epoch)
                    else:
                        txt = 'epoch {:>3} == '.format(epoch)
                else:
                    best_epoch = epoch
                    saver.save(sess, save_path)
                    txt = 'epoch {:>3} >> '.format(epoch)

                # after going through all data sets, do a print out of the current result
                for lk in lossk:
                    txt += '{}: '.format(lk)
                    for dk in data_keys:
                        if len(stat_l3l4[dk][lk]['mean']) > 0:
                            txt += '{:.2f}+-{:.2f}/'.format(stat_l3l4[dk][lk]['mean'][-1],stat_l3l4[dk][lk]['std'][-1])

                    txt = txt[:-1] + ' -- '

                if epoch>0:
                    # print etRange
                    txt = txt + 'similarity: {:.2f}+-{:.2f} -- '.format(stat_et['mean'][-1], stat_et['std'][-1])
                txt = txt + f'time: {(endtime-starttime):.03f}s'
                print(txt)
                with open("result_dpf.txt", 'a') as file0:
                    print(txt,file=file0)

                np.savez(os.path.join(model_path, 'rmse_last'), rmse_last=stat_l3l4)

                if plot:
                    stage['plot'](epoch)

                epoch += 1

            np.savez(os.path.join(model_path, 'exe_time'), log_time=log_time)
            saver.restore(sess, save_path)
        return stat_l3l4

    def predict(self, sess, batch, num_particles, return_particles=False, **kwargs):
        # define input dict, use the first state only if we do tracking
        input_dict = {self.placeholders['o']: batch['o'],
                      self.placeholders['a']: batch['a'],
                      self.placeholders['num_particles']: num_particles}
        if self.init_with_true_state:
            input_dict[self.placeholders['s']] = batch['s'][:, :1]

        if return_particles:
            return sess.run([self.pred_states, self.particle_list, self.particle_probs_list], input_dict)
        else:
            return sess.run(self.pred_states, input_dict)


    def connect_modules(self, means, stds, state_mins, state_maxs, state_step_sizes,std_x, std_y, std_t):

        # get shapes
        self.batch_size = tf.shape(self.placeholders['o'])[0]
        self.seq_len = tf.shape(self.placeholders['o'])[1]
        # we use the static shape here because we need it to build the graph
        self.action_dim = self.placeholders['a'].get_shape()[-1].value

        encodings = snt.BatchApply(self.encoder)((self.placeholders['o'] - means['o']) / stds['o'])
        self.encodings = encodings

        seq_len=self.seq_len

        # initialize particles
        if self.init_with_true_state:
            # tracking with known initial state
            initial_particles = tf.tile(self.placeholders['s'][:, 0, tf.newaxis, :], [1, self.num_particles, 1])
        else:
            # global localization
            # sample particles randomly
            initial_particles = tf.concat(
                    [tf.random_uniform([self.batch_size, self.num_particles, 1], state_mins[d], state_maxs[d]) for d in
                     range(self.state_dim)], axis=-1, name='particles')

        initial_particle_probs = tf.ones([self.batch_size, self.num_particles],
                                         name='particle_probs') / self.num_particles_float

        # assumes that samples has the correct size
        def permute_batch(x, samples):
            # get shapes
            batch_size = tf.shape(x)[0]
            num_particles = tf.shape(x)[1]
            sample_size = tf.shape(samples)[1]
            # compute 1D indices into the 2D array
            idx = samples + num_particles * tf.tile(
                tf.reshape(tf.range(batch_size), [batch_size, 1]),
                [1, sample_size])
            # index using the 1D indices and reshape again
            result = tf.gather(tf.reshape(x, [batch_size * num_particles, -1]), idx)
            result = tf.reshape(result, tf.shape(x[:,:sample_size]))
            return result

        def permute_batch_path(particle_path, noises_path, particle_probs_path, additional_probs_path, samples):
            # get shapes
            batch_size = tf.shape(particle_path)[0]
            num_particles = tf.shape(particle_path)[2]
            state_dims = tf.shape(particle_path)[3]
            sample_size = tf.shape(samples)[1]
            # compute 1D indices into the 2D array
            idx = samples + num_particles * tf.tile(
                tf.reshape(tf.range(batch_size), [batch_size, 1]),
                [1, sample_size])

            particle_path_trans = tf.transpose(particle_path, perm=(0,2,1,3))
            noises_path_trans = tf.transpose(noises_path, perm=(0, 2, 1, 3))

            particle_probs_path_trans = tf.transpose(particle_probs_path, perm=(0, 2, 1))
            additional_probs_path_trans = tf.transpose(additional_probs_path, perm=(0, 2, 1))
            # index using the 1D indices and reshape again
            result = tf.gather(tf.reshape(particle_path_trans, [batch_size * num_particles, -1, state_dims]), idx)
            result = tf.reshape(result, [batch_size, num_particles, -1, state_dims])

            result_n = tf.gather(tf.reshape(noises_path_trans, [batch_size * num_particles, -1, state_dims]), idx)
            result_n = tf.reshape(result_n, [batch_size, num_particles, -1, state_dims])

            result_w = tf.gather(tf.reshape(particle_probs_path_trans, [batch_size * num_particles, -1]), idx)
            result_w = tf.reshape(result_w, [batch_size, num_particles, -1])

            result_wa = tf.gather(tf.reshape(particle_probs_path_trans, [batch_size * num_particles, -1]), idx)
            result_wa = tf.reshape(result_wa, [batch_size, num_particles, -1])
            return tf.transpose(result, perm=(0,2,1,3)), tf.transpose(result_n, perm=(0,2,1,3)), tf.transpose(result_w, perm=(0,2,1)), tf.transpose(result_wa, perm=(0,2,1))


        def loop(particles, particle_probs, particle_list, particle_probs_list, additional_probs_list, etRange_list,noises_list, encoding_input_list,e_t_list, particle_path, noises_path,particle_probs_path, additional_probs_path, i):


            num_resampled_float = self.num_particles_float
            num_resampled = tf.cast(num_resampled_float, tf.int32)

            # resampling
            basic_markers = tf.linspace(0.0, (num_resampled_float - 1.0) / num_resampled_float, num_resampled)
            random_offset = tf.random_uniform([self.batch_size], 0.0, 1.0 / num_resampled_float)
            markers = random_offset[:, None] + basic_markers[None, :]  # shape: batch_size x num_resampled
            cum_probs = tf.cumsum(particle_probs, axis=1)
            marker_matching = markers[:, :, None] < cum_probs[:, None, :]  # shape: batch_size x num_resampled x num_particles
            samples = tf.cast(tf.argmax(tf.cast(marker_matching, 'int32'), axis=2), 'int32')
            standard_particles = permute_batch(particles, samples)
            particle_path, noises_path, particle_probs_path, additional_probs_path = permute_batch_path(particle_path, noises_path, particle_probs_path, additional_probs_path, samples)
            standard_particle_probs = tf.ones([self.batch_size, num_resampled])
            standard_particles = tf.stop_gradient(standard_particles)
            standard_particle_probs = tf.stop_gradient(standard_particle_probs)

            # motion update
            standard_particles, noise_s = self.motion_update(self.placeholders['a'][:, i], standard_particles, means,
                                                             stds, state_step_sizes, std_x, std_y, std_t)
            # measurement update and check encoding_input
            lki, encoding_input, e_t = self.measurement_update(encodings[:, i], standard_particles, means, stds)
            standard_particle_probs *= lki

            etRange_list = tf.cond(tf.equal(i, seq_len - 1),
                                   lambda: lki,
                                   lambda: tf.zeros_like(standard_particle_probs))

            # NORMALIZE AND COMBINE PARTICLES
            particles = standard_particles
            particle_probs = standard_particle_probs

            # NORMALIZE PROBABILITIES
            particle_probs /= tf.reduce_sum(particle_probs, axis=1, keepdims=True)

            particle_list = tf.concat([particle_list, particles[:, tf.newaxis]], axis=1)
            particle_probs_list = tf.concat([particle_probs_list, particle_probs[:, tf.newaxis]], axis=1)
            additional_probs_list = tf.concat([additional_probs_list, lki[:,tf.newaxis]], axis=1)

            particle_path = tf.concat([particle_path, particles[:, tf.newaxis]], axis=1)
            noises_path = tf.concat([noises_path, noise_s[:, tf.newaxis]], axis=1)

            particle_probs_path = tf.concat([particle_probs_path, particle_probs[:, tf.newaxis]], axis=1)
            additional_probs_path = tf.concat([additional_probs_path, lki[:, tf.newaxis]], axis=1)

            noises_list=tf.concat([noises_list, noise_s[:,tf.newaxis,:,:]],axis=1)
            encoding_input_list=tf.concat([encoding_input_list,encoding_input[:,tf.newaxis,:,:]],axis=1)
            e_t_list=tf.concat([e_t_list,e_t[:,tf.newaxis,:,:]],axis=1)

            return particles, particle_probs, particle_list, particle_probs_list, additional_probs_list, etRange_list,noises_list,encoding_input_list,e_t_list, particle_path, noises_path,particle_probs_path, additional_probs_path, i + 1

        particle_list = tf.reshape(initial_particles,
                                   shape=[self.batch_size, -1, self.num_particles, self.state_dim])
        particle_path = tf.reshape(initial_particles,
                                   shape=[self.batch_size, -1, self.num_particles, self.state_dim])
        particle_probs_list = tf.reshape(initial_particle_probs, shape=[self.batch_size, -1, self.num_particles])
        particle_probs_path = tf.reshape(initial_particle_probs, shape=[self.batch_size, -1, self.num_particles])
        additional_probs_list = tf.reshape(tf.ones([self.batch_size, self.num_particles])/ self.num_particles_float, shape=[self.batch_size, -1, self.num_particles])
        additional_probs_path = tf.reshape(tf.ones([self.batch_size, self.num_particles]) / self.num_particles_float,
                                           shape=[self.batch_size, -1, self.num_particles])
        etRange_list=tf.zeros([self.batch_size,self.num_particles])

        noises_list=tf.reshape(tf.zeros([self.batch_size,self.num_particles,self.state_dim]),shape=[self.batch_size, -1, self.num_particles, self.state_dim])
        noises_path = tf.reshape(tf.zeros([self.batch_size, self.num_particles, self.state_dim]),
                                 shape=[self.batch_size, -1, self.num_particles, self.state_dim])

        encoding_input_list=tf.reshape(tf.zeros([self.batch_size,self.num_particles,self.hidden_size]),shape=[self.batch_size, -1, self.num_particles, self.hidden_size])
        e_t_list=tf.reshape(tf.zeros([self.batch_size,self.num_particles,self.hidden_size]),shape=[self.batch_size, -1, self.num_particles,self.hidden_size])

        # run the filtering process
        particles, particle_probs, particle_list, particle_probs_list, additional_probs_list, etRange_list, noises_list, encoding_input_list, e_t_list, particle_path, noises_path, particle_probs_path, additional_probs_path, i = tf.while_loop(
            lambda *x: x[-1] < self.seq_len, loop,
            [initial_particles, initial_particle_probs, particle_list, particle_probs_list, additional_probs_list,
             etRange_list, noises_list, encoding_input_list, e_t_list, particle_path, noises_path, particle_probs_path,
             additional_probs_path,
             tf.constant(1, dtype='int32')], name='loop')

        # compute mean of particles
        self.pred_states = self.particles_to_state(particle_list, particle_probs_list)
        self.particle_list = particle_list
        self.particle_probs_list = particle_probs_list

        return particles, particle_probs, encodings, particle_list, particle_probs_list, additional_probs_list, etRange_list, noises_list, encoding_input_list, e_t_list, particle_path, noises_path, particle_probs_path, additional_probs_path

    def particles_to_state(self, particle_list, particle_probs_list):
        mean_position = tf.reduce_sum(particle_probs_list[:, :, :, tf.newaxis] * particle_list[:, :, :, :2], axis=2)
        mean_orientation = atan2(
            tf.reduce_sum(particle_probs_list[:, :, :, tf.newaxis] * tf.cos(particle_list[:, :, :, 2:]), axis=2),
            tf.reduce_sum(particle_probs_list[:, :, :, tf.newaxis] * tf.sin(particle_list[:, :, :, 2:]), axis=2))
        return tf.concat([mean_position, mean_orientation], axis=2)


    def plot_motion_model(self, sess, batch, motion_samples, task):

        # define the inputs and train/run the model
        input_dict = {**{self.placeholders[key]: batch[key] for key in 'osa'},
                      **{self.placeholders['num_particles']: 100},
                      }

        s_motion_samples = sess.run(motion_samples, input_dict)

        plt.figure('Motion Model')
        plt.gca().clear()
        plot_maze(task)
        for i in range(min(len(s_motion_samples), 10)):
            plt.quiver(s_motion_samples[i, :, 0], s_motion_samples[i, :, 1], np.cos(s_motion_samples[i, :, 2]), np.sin(s_motion_samples[i, :, 2]), color='blue', width=0.001, scale=100)
            plt.quiver(batch['s'][i, 0, 0], batch['s'][i, 0, 1], np.cos(batch['s'][i, 0, 2]), np.sin(batch['s'][i, 0, 2]), color='black', scale=50, width=0.003)
            plt.quiver(batch['s'][i, 1, 0], batch['s'][i, 1, 1], np.cos(batch['s'][i, 1, 2]), np.sin(batch['s'][i, 1, 2]), color='red', scale=50, width=0.003)

        plt.gca().set_aspect('equal')
        plt.pause(0.01)


    def plot_measurement_model(self, sess, batch_iterator, measurement_model_out):

        batch = next(batch_iterator)
        # define the inputs and train/run the model
        input_dict = {**{self.placeholders[key]: batch[key] for key in 'osa'},
                      **{self.placeholders['num_particles']: 100},
                      }

        s_measurement_model_out = sess.run(measurement_model_out, input_dict)

        plt.figure('Measurement Model Output')
        plt.gca().clear()
        plt.imshow(s_measurement_model_out, interpolation="nearest", cmap="coolwarm")
        plt.pause(0.01)


    def plot_particle_proposer(self, sess, batch, proposed_particles, task):

        # define the inputs and train/run the model
        input_dict = {**{self.placeholders[key]: batch[key] for key in 'osa'},
                      **{self.placeholders['num_particles']: 100},
                      }

        s_samples = sess.run(proposed_particles, input_dict)

        plt.figure('Particle Proposer')
        plt.gca().clear()
        plot_maze(task)

        for i in range(min(len(s_samples), 10)):
            color = np.random.uniform(0.0, 1.0, 3)
            plt.quiver(s_samples[i, :, 0], s_samples[i, :, 1], np.cos(s_samples[i, :, 2]), np.sin(s_samples[i, :, 2]), color=color, width=0.001, scale=100)
            plt.quiver(batch['s'][i, 0, 0], batch['s'][i, 0, 1], np.cos(batch['s'][i, 0, 2]), np.sin(batch['s'][i, 0, 2]), color=color, scale=50, width=0.003)

        plt.pause(0.01)


    def plot_particle_filter(self, sess, batch, task,epoch):

        num_particles = 1000
        head_scale = 1.5
        quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 40., 'width': 0.003, 'headlength': 5 * head_scale,
                       'headwidth': 3 * head_scale, 'headaxislength': 4.5 * head_scale}
        marker_kwargs = {'markersize': 4.5, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}

        color_list = plt.cm.tab10(np.linspace(0, 1, 10))
        colors = {'lstm': color_list[0], 'pf_e2e': color_list[1], 'pf_ind_e2e': color_list[2], 'pf_ind': color_list[3],
                  'ff': color_list[4], 'odom': color_list[4]}

        pred, s_particle_list, s_particle_probs_list = self.predict(sess, batch, num_particles,
                                                                      return_particles=True)
        num_steps = 100

        for s in range(1):

            plt.figure("example {}".format(s), figsize=[12, 5.15])
            plt.gca().clear()
            i1=0

            for i in range(num_steps-20,num_steps):
                ax = plt.subplot(4, 5, i1 + 1, frameon=False)
                # ax = plt.subplot(10, 10, i + 1, frameon=False)
                plt.gca().clear()

                plot_maze(task, margin=5, linewidth=0.5)

                if i < num_steps - 1:
                    ax.quiver(s_particle_list[s, i, :, 0], s_particle_list[s, i, :, 1],
                              np.cos(s_particle_list[s, i, :, 2]), np.sin(s_particle_list[s, i, :, 2]),
                              s_particle_probs_list[s, i, :], cmap='viridis_r', clim=[.0, 2.0 / num_particles],
                              alpha=1.0,
                              **quiv_kwargs
                              )

                    current_state = batch['s'][s, i, :]
                    plt.quiver(current_state[0], current_state[1], np.cos(current_state[2]),
                               np.sin(current_state[2]), color="red", **quiv_kwargs)

                    plt.plot(current_state[0], current_state[1], 'or', **marker_kwargs)
                else:

                    ax.plot(batch['s'][s, :num_steps, 0], batch['s'][s, :num_steps, 1], '-', linewidth=0.6, color='red')
                    ax.plot(pred[s, :num_steps, 0], pred[s, :num_steps, 1], '-', linewidth=0.6,
                            color=colors['pf_ind_e2e'])

                    ax.plot(batch['s'][s, :1, 0], batch['s'][s, :1, 1], '.', linewidth=0.6, color='red', markersize=3)
                    ax.plot(pred[s, :1, 0], pred[s, :1, 1], '.', linewidth=0.6, markersize=3,
                            color=colors['pf_ind_e2e'])

                plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.001, hspace=0.1)
                plt.gca().set_aspect('equal')
                plt.xticks([])
                plt.yticks([])
                i1=i1+1

        plt.savefig('../plots/test {}.pdf'.format(epoch), transparent=True, dpi=600, facecolor='w',
                    pad_inches=0.01)
        show_pause(pause=0.01)

