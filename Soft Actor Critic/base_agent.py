import tensorflow as tf
import numpy as np

class Base_Agent(): 

    def __init__(self,**params):

        self.log_dir = params.get('log_dir',False)
        self.logging = not(self.log_dir is False)
        self.train_steps = 0

        self.train_ops = []

    def _finish_agent_setup(self):
        if self.logging:
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_session().graph)
            self.merged = tf.summary.merge_all()
        tf.global_variables_initializer().run() 

    def _construct_feed_dict(self,samples):  
        raise NotImplementedError
                    
    def _train(self, samples, *args):   
        assert len(args) == 1

        feed_dict = self._construct_feed_dict(samples)  
        if self.logging:
            _,summary,returns =  tf.get_default_session().run([self.train_ops,self.merged, args[0]], feed_dict = feed_dict)
            self.writer.add_summary(summary,self.train_steps)
            return returns
        else:
            _,returns =  tf.get_default_session().run([self.train_ops, args[0]], feed_dict = feed_dict)
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


