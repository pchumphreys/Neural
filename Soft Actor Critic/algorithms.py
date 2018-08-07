import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Deep_Q_Learning(Agent):
    # This class handles the training of the networks
    def __init__(self,n_inputs,n_outputs,**params):

        super(Deep_Q_Learning,self).__init__(n_inputs,n_outputs,**params)
      
        self.lr = params['agent'].pop('lr',1e-3)
        self.discount = params['agent'].pop('discount',1e-3)
        self.tau = params['agent'].pop('tau',1e-3)
        self.dueling =  params['agent'].pop('dueling', True)
        self.huber_loss =  params['agent'].pop('huber_loss', True)
        self.clip_gradients =  params['agent'].pop('clip_gradients', False)
        self.train_steps_per_t = params['agent'].pop('train_steps_per_t',1)


        self.Qnet = Qnet(n_outputs,self.observations,params['network_spec'])
        self.model_Q_params = self.Qnet.get_params_internal()
        self.model_Q_outputs = Qnet.output
        self.model_Qnet_predict = tf.stop_gradient(tf.one_hot(tf.argmax(Qnet.make_network(inputs = self.next_obs),axis=1),self.Qnet.output_size))
        
        # Duplicate the Qnet with different variables for the target network
        with tf.variable_scope('qNet_T'):
            self.target_Q_outputs = Qnet.make_network(inputs = self.next_obs,reuse=False) 
            self.target_Q_params = Qnet.get_params_internal()
        
        self.policy = Policy_Discrete(self.Qnet,**params['policy'])

        self.rb = Replay_Buffer(n_inputs,n_outputs,**params['replay_buffer'])
    
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        
        self.train_ops = []
        self.init_Q_net_training()
        self.init_target_Q_update()

        self._finish_agent_setup()
        

    def init_Q_net_training(self):
        

        with tf.variable_scope('Q_loss'):

            if not(self.dueling):
                self.Q_t = tf.stop_gradient(self.rewards +  self.discount * (1-self.dones) * tf.reduce_max(self.target_Q_outputs,axis=1))
            else:
                self.Q_t = tf.stop_gradient(self.rewards +  self.discount * (1-self.dones) * tf.reduce_sum(self.target_Q_outputs*self.Qnet_predict,axis=1))
        
            if self.huber_loss:
                self.Q_Loss = tf.reduce_mean(tf.sqrt(1+tf.square(tf.reduce_sum(self.Q_outputs*self.actions,axis=1) - self.Q_t))-1)
            else:
                self.Q_Loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.Q_outputs*self.actions,axis=1) - self.Q_t))

            tf.summary.scalar('Q_loss', self.Q_Loss)
        

        Qnet_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self.Qnet._name)
        Qnet_regularization_loss = tf.reduce_sum(
            Qnet_regularization_losses)
    
        gradients, variables = zip(*self.optimizer.compute_gradients(self.Q_Loss + Qnet_regularization_loss,var_list = training_variables = self.model_Q_params))
        if self.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
        self.train_Q = self.optimizer.apply_gradients(zip(gradients, variables))

        self.train_ops.append(self.train_Q)
        
    def init_target_Q_update(self):
        
        with tf.variable_scope('Target_Q_update'):
            self.tQnet_update = []
            for tQ_p in self.target_Q_params:
                # Match each target net param with equiv from vnet
                Q_p = [v for v in self.model_Q_params if tQ_p.name[(tQ_p.name.index('/')+1):] in v.name]
                assert(len(Q_p) == 1) # Check that only found one variable
                Q_p = Q_p[0]
                with tf.control_dependencies([self.train_Q]):
                    self.tQnet_update.append(tQ_p.assign(self.tau * Q_p + (1-self.tau)*tQ_p))
            self.tQnet_update = tf.group(self.tQnet_update)
            
        self.train_ops.append(self.tQnet_update)

    def train(self):
        if self.rb.batch_ready():
            for j in range(self.train_steps_per_t):
                samples = self.rb.get_samples()
                summary,losses = self._train(samples,self.Q_Loss)
        else:
            losses = False
        return losses

    def add_sample(self,action,current_obs,next_obs,reward,done):
        self.rb.add_sample(action,current_obs,next_obs,reward,done)

             

class Agent(): 

    def __init__(self,n_inputs,n_outputs,**params):

        self.log_dir = params.pop['log_dir',False]

        self.actions = tf.placeholder(tf.float32,shape = [None,n_outputs],name = 'actions')
        self.obs = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'observations')
        self.next_obs = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'next_observations')
        self.rewards = tf.placeholder(tf.float32,shape = [None],name = 'rewards')
        self.dones = tf.placeholder(tf.float32,shape = [None],name = 'dones')

        self.train_steps = 0

        self.train_ops = []

    def _finish_agent_setup(self):
        if self.log_dir:
            self.writer = tf.summary.FileWriter(log_dir, tf.get_default_session().graph)
            self.merged = tf.summary.merge_all()
        tf.global_variables_initializer().run() 

    def _construct_feed_dict(self,samples):  
        return {self.actions : samples['actions'],
                    self.obs : samples['observations'],
                    self.next_obs : samples['next_observations'],
                    self.dones : samples['dones'],
                    self.rewards : samples['rewards']}
                    
    def _train(self, samples, *args):     

        feed_dict = self._construct_feed_dict(samples)
        if self.log_dir:
            _,summary,returns =  tf.get_default_session().run([self.train_ops] + self.merged + list(args), feed_dict = feed_dict)
            self.writer.add_summary(summary,self.train_steps)
            return returns
        else:
            _,returns =  tf.get_default_session().run([self.train_ops]+ list(args), feed_dict = feed_dict)
            return returns

        self.train_steps += 1

    def episode_done(self):
        if self.log_dir:
            self.writer.flush()

    def train(self,*kw):
        raise NotImplementedError

    def add_sample(self,*kw):
        raise NotImplementedError


