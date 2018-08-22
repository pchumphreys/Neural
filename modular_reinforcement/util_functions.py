import tensorflow as tf
import numpy as np
from scipy.signal import lfilter
import os
import cv2
def update_target_network(ref_net_params,target_net_params,tau=1.0,update_op_control_dependencies=[]):
    if not(isinstance(update_op_control_dependencies,list)):
        update_op_control_dependencies = [update_op_control_dependencies]
        
    target_update_ops = []
    for t_p in target_net_params:
        # Match each target net param with equiv from reference net
        r_p = [v for v in ref_net_params if t_p.name[(t_p.name.index('/')+1):] in v.name[(v.name.index('/')+1):]]
        assert(len(r_p) == 1) # Check that only found one variable
        r_p = r_p[0]
        with tf.control_dependencies(update_op_control_dependencies):
            if tau == 'Discrete':
                target_update_ops.append(t_p.assign(r_p))
            else:
                target_update_ops.append(t_p.assign(tau * r_p + (1-tau)*t_p))

    return tf.group(target_update_ops)

def calc_discount(signal,discount,axis=0):
    return np.flip(lfilter([1], [1, -discount], np.flip(signal,axis), axis=axis),axis)
    
def mask_rewards_using_dones(dones,axis=1): # Use dones to make a mask -> if done, all next steps should be False
    return (lfilter([0,1], [1, -1], (dones.astype(float)), axis=axis) < 1)


def aws_save_to_bucket(source_dir,target_file_name):
    import shutil
    import boto3
    s3 = boto3.resource('s3')
    zipf = shutil.make_archive(target_file_name, 'zip', source_dir)
    s3.Bucket('phumphreys-neural').upload_file(zipf,os.path.join('run_logs',target_file_name))

def preprocess_image_obs(observation):
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation.astype(np.bool_),(84,84,1))