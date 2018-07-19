import numpy as np
import tensorflow as tf
import scipy.signal
from sklearn.utils import shuffle

# Value function 
class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, hid1_mult):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
            hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()


# Policy function
class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, clipping_range=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            policy_logvar: natural log of initial policy variance
        """
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.epochs = 20
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clipping_range = clipping_range
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        self.means = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)), name="means")
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        if self.clipping_range is not None:
            print('setting up loss with clipping objective')
            pg_ratio = tf.exp(self.logp - self.logp_old)
            clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.clipping_range[0], 1 + self.clipping_range[1])
            surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                        self.advantages_ph * clipped_pg_ratio)
            self.loss = -tf.reduce_mean(surrogate_loss)
        else:
            print('setting up loss with KL penalty') # <= We use this 
            loss1 = -tf.reduce_mean(self.advantages_ph *
                                    tf.exp(self.logp - self.logp_old))
            loss2 = tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0*self.kl_targ))
            self.loss = loss1 + loss2 + loss3
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset_model()
    for _ in range(10): 
        obs,_,_,_ = env.step(np.zeros(shape=env.actDim))
    env.sim.data.time = 0
    observes, actions, rewards, unscaled_obs, rDetails = \
        [], [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    tick = 0
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1)) # Make rank two
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        # Normalize observations
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        # Sample action and make it rank two 
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        # Forward step 
        obs, reward, done, rDetail = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            rwd_before = reward
            rwd_after = np.asscalar(reward)
            reward = rwd_after
            print ('rwd: %s => %s'%(rwd_before,rwd_after))
        rewards.append(reward)
        rDetails.append(rDetail)
        step += env.dt  # increment time step feature
        tick += 1
        sec = env.sim.data.time # Get time [sec]
        # print ("tick:[%d] step:[%.2f] sec:[%.2fsec]"%(tick,step,sec))
        if sec > 9.0:
            done = True
    # Return (scaled) observations, actions, rewards, and unscaled obersavtions 
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs),
            rDetails)

def run_episode_vid(env,policy,scaler):
    obs = env.reset_model()
    for _ in range(10): 
        obs,_,_,_ = env.step(np.zeros(shape=env.actDim))
    env.sim.data.time = 0
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0 
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    tick = 0
    frames = []
    rewards = []
    time_out = False
    reward_sum = 0
    xInit = env.get_body_com("torso")[0]
    hInit = env.get_heading()
    while not time_out:
        frames.append(env.render(mode='rgb_array',width=200,height=200))
        obs = obs.astype(np.float32).reshape((1, -1)) # Make rank two
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        # Normalize observations
        obs = (obs - offset) * scale  # center and scale observations
        # Sample action and make it rank two 
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        # Forward step 
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        rewards.append(reward)
        reward_sum += reward
        tick += 1
        step += env.dt  # increment time step feature
        sec = env.sim.data.time # Get time [sec]
        if sec > 9.0:
            time_out = True
    xFinal = env.get_body_com("torso")[0]
    hFinal = env.get_heading()
    xDisp = xFinal - xInit
    hDisp = hFinal - hInit
    ret = {'frames':frames,'xDisp':xDisp,'hDisp':hDisp,'rewards':rewards}
    return ret     

def run_policy(env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run (=#trajectory)

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = [] # Empty list 
    for e in range(episodes):
        # print ("epi:[%d/%d]"%(e,episodes))
        observes, actions, rewards, unscaled_obs, rDetails =\
            run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs,
                      'rDetails':rDetails}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})
    return trajectories

def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes) # Predict value
        trajectory['values'] = values

def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point 
        # Following mimics this operation
        _x = np.array([1,1,1,1,1,1,1,1,1])
        _gamma = 0.95
        print (_x)
        print (discount(_x,_gamma))
        temp = 0
        for i in range(9):
            temp += 1*_gamma**i
            print ("[%d] val:[%.2f]"%(i,temp))
    """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories
    Args:
        trajectories: as returned by run_policy()
        gamma: discount
    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories: # For all trajs,
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew

def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values'] # From the critic network 
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        # Now we have advatages 
        trajectory['advantages'] = advantages

def build_train_set(trajectories):
    """
    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    
    # This will be used for updating value net
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    
    # normalize advantages (this is heuristic post processing..)
    # This will be used for updating policy 
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })

