import tensorflow as tf
import numpy as np

class Agent(): 

    def __init__(self,n_inputs,n_outputs,**params):

        self.log_dir = params.pop('log_dir',False)
        self.logging = not(self.log_dir is False)

        self.actions = tf.placeholder(tf.float32,shape = [None,n_outputs],name = 'actions')
        self.obs = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'observations')
        self.next_obs = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'next_observations')
        self.rewards = tf.placeholder(tf.float32,shape = [None],name = 'rewards')
        self.dones = tf.placeholder(tf.float32,shape = [None],name = 'dones')

        self.train_steps = 0

        self.train_ops = []

    def _finish_agent_setup(self):
        if self.logging:
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_session().graph)
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
        if self.logging:
            _,summary,returns =  tf.get_default_session().run([self.train_ops,self.merged] + list(args), feed_dict = feed_dict)
            self.writer.add_summary(summary,self.train_steps)
            return returns
        else:
            _,returns =  tf.get_default_session().run([self.train_ops]+ list(args), feed_dict = feed_dict)
            return returns

        self.train_steps += 1

    def episode_finished(self):
        if self.logging:
            self.writer.flush()

    def get_action(self,*kw):
        raise NotImplementedError

    def train(self,*kw):
        raise NotImplementedError

    def add_sample(self,*kw):
        raise NotImplementedError


