import tensorflow as tf
from scipy.signal import lfilter
def update_target_network(ref_net_params,target_net_params,tau=1.0,update_op_control_dependencies=[]):

    target_update_ops = []
    for t_p in target_net_params:
        # Match each target net param with equiv from reference net
        r_p = [v for v in ref_net_params if t_p.name[(t_p.name.index('/')+1):] in v.name[(v.name.index('/')+1):]]
        assert(len(r_p) == 1) # Check that only found one variable
        r_p = r_p[0]
        with tf.control_dependencies(update_op_control_dependencies):
            target_update_ops.append(t_p.assign(tau * r_p + (1-tau)*t_p))

    return tf.group(target_update_ops)

def calc_discount(signal,discount):
    return lfilter([1], [1, -discount], signal[::-1], axis=0)[::-1]
