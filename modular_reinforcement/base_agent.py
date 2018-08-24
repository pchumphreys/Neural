import tensorflow as tf
import numpy as np

class Base_Agent(): 

    def __init__(self,**params):

        self.log_dir = params.get('log_dir',False)
        self.logging = not(self.log_dir is False)
        self.train_steps = 0
        self.sess = tf.get_default_session()
        self.train_ops = []

    def _finish_agent_setup(self):
        if self.logging:
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_session().graph)
            self.merged = tf.summary.merge_all()
        tf.global_variables_initializer().run() 

    def _construct_feed_dict(self,samples):  
        raise NotImplementedError

    def _get_regs_add_clip_make_optimizer(self,params,loss,scope = None):

        regularization_loss = tf.reduce_sum(tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=scope))

        gradients, variables = zip(*self.optimizer.compute_gradients(loss + regularization_loss,var_list = params))
        if self.clip_gradients:
            gradients, grad_norms = tf.clip_by_global_norm(gradients, self.clip_gradients)
        tf.summary.scalar('grad_norms', grad_norms)
        return self.optimizer.apply_gradients(zip(gradients, variables))

                    
    def _train(self, samples,global_step, *args):   
        assert len(args) == 1

        feed_dict = self._construct_feed_dict(samples)  
        if self.logging:
            _,summary,returns =  self.sess.run([self.train_ops,self.merged, args[0]], feed_dict = feed_dict)
            if self.train_steps % 100 == 0:
                self.writer.add_summary(summary,global_step)

        else:
            _,returns =  self.sess.run([self.train_ops, args[0]], feed_dict = feed_dict)
      

        self.train_steps += 1
        return returns

    def episode_finished(self,episode_reward,global_step):
        if self.logging:
            summary = tf.Summary(value=[
                    tf.Summary.Value(tag="Episode rewards", simple_value=episode_reward)])
            self.writer.add_summary(summary,global_step)
            self.writer.flush()
            
     
    
    def get_action(self,obs,optimal_action = False):

        return self.policy.get_actions(np.asarray(obs),optimal_action = optimal_action)


    def train(self,*kw):
        raise NotImplementedError

    def add_sample(self,*kw):
        raise NotImplementedError


