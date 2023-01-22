import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import numpy as np
from time import time
import random
from scipy.interpolate import interp1d

import h5py
import matplotlib.pyplot as plt
from glob import glob
import math
import os
import cupy as cp

class DRL(object):
    def __init__(self, place_holder=''):
        self.place_holder = place_holder

    def model_inputs(self, height, width, channels):
        """
        Create the model inputs/tensors
        """        
        obs_pre = tf.placeholder(tf.float32, (None, None, None, None, 10), name='obs_pre')
        state_pre = tf.placeholder(tf.float32, (None, None, None, None, 10), name='state_pre')
        state_stat_pre = tf.placeholder(tf.float32, (None, None, None, None, 9), name='state_stat_pre')     

        action = tf.placeholder(tf.float32, (None, None, None, None, 6), name='action')
        reward = tf.placeholder(tf.float32, (None, None, None, None, 1), name='reward')        

        state_action_mean = tf.placeholder(tf.float32, (None, 16+9), name='state_action_mean')
        state_action_std = tf.placeholder(tf.float32, (None, 16+9), name='state_action_std')

        critic_learning_rate = tf.placeholder(tf.float32, [], name='critic_learning_rate')
        actor_learning_rate = tf.placeholder(tf.float32, [], name='actor_learning_rate')

        SL_input = tf.placeholder(tf.float32, (None, None, None, None, 10), name='SL_input')
        SL_target = tf.placeholder(tf.float32, (None, None, None, None, 6), name='SL_target')
        SL_weight = tf.placeholder(tf.float32, (None, None, None, None, 6), name='SL_weight')

        return state_pre, state_stat_pre, obs_pre, action, reward, state_action_mean, state_action_std, \
               critic_learning_rate, actor_learning_rate, \
               SL_input, SL_target, SL_weight

            
    def periodic_padding2D(self, image, padding=1):
        #periodic padding
        left_pad = image[:,:,-padding:,:]
        right_pad = image[:,:,:padding,:]
            
        partial_image = tf.concat([left_pad, image, right_pad], axis=2)
            
        #zero padding
        upper_pad = partial_image[:,-padding:,:,:]
        lower_pad = partial_image[:,:padding,:,:]
            
        padded_image = tf.concat([upper_pad, partial_image, lower_pad], axis=1)

        return padded_image


    def get_weight(self, shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
        if fan_in is None: fan_in = np.prod(shape[:-1])
        std = gain / np.sqrt(fan_in) # He init
        if use_wscale:
            wscale = tf.constant(np.float32(std), name='wscale')
            return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
        else:
            return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

    def get_weight_final(self, shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
        if fan_in is None: fan_in = np.prod(shape[:-1])

        return tf.get_variable('weight', shape=shape, initializer=tf.random_uniform_initializer(minval=-0.1/fan_in, maxval=0.1/fan_in))


    #----------------------------------------------------------------------------
    # Convolutional layer.
    def conv2d(self, x, fmaps, kernel, strides, gain=np.sqrt(2), use_wscale=False):
        assert kernel >= 1 and kernel % 2 == 1
        w = self.get_weight([kernel, kernel, x.shape[3].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)
        if kernel >= 3:
            x = self.periodic_padding2D(x, padding = (kernel-1)//2)
        return tf.nn.conv2d(x, w, strides=[1,strides,strides,1], padding='VALID', data_format='NHWC')

    def conv2d_final(self, x, fmaps, kernel, strides, gain=np.sqrt(2), use_wscale=False):
        assert kernel >= 1 and kernel % 2 == 1
        w = self.get_weight_final([kernel, kernel, x.shape[3].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)
        if kernel >= 3:
            x = self.periodic_padding2D(x, padding = (kernel-1)//2)
        return tf.nn.conv2d(x, w, strides=[1,strides,strides,1], padding='VALID', data_format='NHWC')

    #----------------------------------------------------------------------------
    # Apply bias to the given activation tensor.
    def apply_bias(self, x):
        if len(x.shape) == 2:
            b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
            b = tf.cast(b, x.dtype)
            return x + b
        if len(x.shape) == 4:
            b = tf.get_variable('bias', shape=[x.shape[3]], initializer=tf.initializers.zeros())
            b = tf.cast(b, x.dtype)
            return x + tf.reshape(b, [1, 1, 1, -1])
        if len(x.shape) == 5:
            b = tf.get_variable('bias', shape=[x.shape[4]], initializer=tf.initializers.zeros())
            b = tf.cast(b, x.dtype)
            return x + tf.reshape(b, [1, 1, 1, 1, -1])

            
    #----------------------------------------------------------------------------
    # Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.
    def act(self, x, alpha=0.2):
        #with tf.name_scope('Tanh'):
        #    return tf.nn.tanh(x)
        #with tf.name_scope('Relu'):
        #    return tf.nn.relu(x)
        with tf.name_scope('LeakyRelu'):
            alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
            return tf.maximum(x * alpha, x)

    #----------------------------------------------------------------------------



    def critic(self, NET_name, input_x1, input_x2, state_action_mean, state_action_std, reuse=False):

        with tf.variable_scope(NET_name, reuse=reuse):

            use_wscale = False

            input_x1 = tf.reshape(input_x1, [-1, 1, nzp_rl, nxp_rl, 10+6])
            input_x2 = tf.reshape(input_x2, [-1, 1, nzp_rl, nxp_rl, 9])
            input_x = tf.concat([input_x1, input_x2], axis=4)

            input_x = tf.reshape(input_x, [-1, nzp_rl, nxp_rl, 16+9])
            input_x_mean = tf.reshape(state_action_mean, [-1, 1, 1, 16+9])
            input_x_std = tf.reshape(state_action_std, [-1, 1, 1, 16+9])
            input_x = (input_x-input_x_mean) / (input_x_std + 10**-10)  #(input_x) / (input_x_std + 10**-10)  #        


            y_sym = np.reshape([1., -1., 1., -1., 1., -1., 1., -1., 1., 1., 1., 1., 1., -1., 1., -1., 1., 1., -1., -1., 1., 1., 1., -1., 1.], [1,1,1,16+9])
            z_sym = np.reshape([1., 1., -1., 1., 1., -1., -1., -1., 1., 1., 1., 1., 1., 1., -1., -1., 1., -1., 1., -1., 1., 1., 1., 1., 1.], [1,1,1,16+9])

            with tf.variable_scope('state-action'):

                #x = tf.concat([h1,h2], axis=3)
                #x = input_x
                x = tf.concat([input_x[:,:,:,:], z_sym*input_x[:,::-1,:,:], y_sym*input_x[:,:,:,:], z_sym*y_sym*input_x[:,::-1,:,:]], axis=0)
                with tf.variable_scope('Stat-Conv0-%d'%0):
                    x = self.act(self.apply_bias(self.conv2d(x, 64, 3, 1, use_wscale=use_wscale)))
                with tf.variable_scope('Stat-Conv0-%d'%1):
                    x = self.act(self.apply_bias(self.conv2d(x, 128, 3, 1, use_wscale=use_wscale)))
                x = tf.nn.avg_pool2d(x, [1,2,2,1], [1,2,2,1], padding='VALID', data_format='NHWC') #8->4
                with tf.variable_scope('Stat-Conv1-%d'%0):
                    x = self.act(self.apply_bias(self.conv2d(x, 128, 3, 1, use_wscale=use_wscale)))
                with tf.variable_scope('Stat-Conv1-%d'%1):
                    x = self.act(self.apply_bias(self.conv2d(x, 256, 3, 1, use_wscale=use_wscale)))
                x = tf.nn.avg_pool2d(x, [1,2,2,1], [1,2,2,1], padding='VALID', data_format='NHWC') #4->2
                with tf.variable_scope('Stat-Conv2-%d'%0):
                    x = self.act(self.apply_bias(self.conv2d(x, 256, 3, 1, use_wscale=use_wscale)))
                with tf.variable_scope('Stat-Conv2-%d'%1):
                    x = self.act(self.apply_bias(self.conv2d(x, 256, 3, 1, use_wscale=use_wscale)))
                x = tf.nn.avg_pool2d(x, [1,2,2,1], [1,2,2,1], padding='VALID', data_format='NHWC') #2->1
                with tf.variable_scope('Dense0'):
                    x = tf.reshape(x, [-1, 1, 1, 1*1*256])
                    x = self.act(self.apply_bias(self.conv2d(x, 256, 1, 1, use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = self.act(self.apply_bias(self.conv2d(x, 256, 1, 1, use_wscale=use_wscale)))
                with tf.variable_scope('Dense2'):
                    x = self.apply_bias(self.conv2d(x, 1, 1, 1, gain=1.0, use_wscale=use_wscale)) 
                    x_original, x_z, x_y, x_yz = tf.split(x, 4, axis=0)

            img = 0.25*(x_original + x_z + x_y + x_yz)

        return tf.reshape(img, [-1,1])
    
    def actor(self, NET_name, input_x, reuse=False):

        with tf.variable_scope(NET_name, reuse=reuse):
            fmaps = [128, 128, 64, 64, 32, 32, 16, 16]
            use_wscale = False
            input_x = tf.reshape(input_x, [-1, 1, 1, 10])#[:,:,:,0:3:2]

            x0 = input_x
            TAU_bc = (x0[:,:,:,0:1]**2 + x0[:,:,:,1:2]**2 + x0[:,:,:,2:3]**2 + x0[:,:,:,4:5]**2 + x0[:,:,:,6:7]**2 + x0[:,:,:,7:8]**2 + x0[:,:,:,8:9]**2)**0.5

            state_z_sym = tf.reshape([1., 1., -1., 1., 1., -1., -1., -1., 1.], [1, 1, 1, 9])
            action_z_sym = tf.reshape([1., 1., 1., 1., -1., -1.], [1, 1, 1, 6])
            state_y_sym = tf.reshape([1., -1., 1., -1., 1., -1., 1., -1., 1.], [1, 1, 1, 9])
            action_y_sym = tf.reshape([1., 1., 1., -1., 1., -1.], [1, 1, 1, 6])
            with tf.variable_scope('TAU'):
                x = tf.concat([x0[:,:,:,0:9], state_z_sym*x0[:,:,:,0:9], state_y_sym*x0[:,:,:,0:9], state_y_sym*state_z_sym*x0[:,:,:,0:9]], axis=0)
                #x = x0[:,:,:,0:9]#
                with tf.variable_scope('Conv0-%d'%0):
                    x = self.act(self.apply_bias(self.conv2d(x, 128, 1, 1, use_wscale=use_wscale)))
                with tf.variable_scope('Conv0-%d'%1):
                    x = self.act(self.apply_bias(self.conv2d(x, 128, 1, 1, use_wscale=use_wscale))) 
                with tf.variable_scope('Conv0-%d'%2):
                    x = self.act(self.apply_bias(self.conv2d(x, 128, 1, 1, use_wscale=use_wscale))) 
                with tf.variable_scope('Conv0-%d'%3):
                    x = self.act(self.apply_bias(self.conv2d(x, 128, 1, 1, use_wscale=use_wscale))) 
                with tf.variable_scope('Conv0-%d'%4):
                    x = self.act(self.apply_bias(self.conv2d(x, 128, 1, 1, use_wscale=use_wscale))) 
                with tf.variable_scope('Conv0-%d'%5):
                    x = self.act(self.apply_bias(self.conv2d(x, 128, 1, 1, use_wscale=use_wscale))) 
                with tf.variable_scope('Conv0-%d'%6):
                    x = 0.00001*(self.apply_bias(self.conv2d(x, 6, 1, 1, use_wscale=use_wscale)))#/tf.reduce_sum(input_x**2, axis=3, keepdims=True)**0.5
                    x = tf.reshape(x, [-1, 1, 1, 6])
            x_original, x_z, x_y, x_yz = tf.split(x, 4, axis=0)
            x = x0[:,:,:,9:10]*TAU_bc*0.25*(x_original + action_z_sym*x_z + action_y_sym*x_y + action_z_sym*action_y_sym*x_yz)


        return tf.reshape(x, [-1,6])


    def model_train(self, state_pre, state_stat_pre, obs_pre, action, reward, 
                    critic_learning_rate, actor_learning_rate, SL_input, SL_target, SL_weight):

        state_pre = tf.reshape(state_pre, [-1, 1, nzp_rl, nxp_rl, 10])
        state_stat_pre = tf.reshape(state_stat_pre, [-1, 1, nzp_rl, nxp_rl, 9])
        obs_pre = tf.reshape(obs_pre, [-1, 1, nzp_rl, nxp_rl, 10])
        action = tf.reshape(action, [-1, 1, nzp_rl, nxp_rl, 6])
        reward = tf.reshape(reward, [-1, 1, 1])

        critic_input_action = tf.concat([state_pre, action],axis=4)
        Q_predict1 = tf.reshape(self.critic('critic1', critic_input_action, state_stat_pre, state_action_mean, state_action_std, reuse=False), [-1, 1, 1])
        TD_loss1 = tf.reduce_mean(tf.abs(Q_predict1-reward)**2)
        
        action_pre = tf.reshape(self.actor('actor',obs_pre, reuse=False), [-1, 1, nzp_rl, nxp_rl, 6])
        critic_input_action_pre = tf.concat([state_pre, action_pre],axis=4)
        Q_loss = tf.reduce_mean(self.critic('critic1', critic_input_action_pre, state_stat_pre, state_action_mean, state_action_std, reuse=True))
        action_MS = tf.reduce_mean(action_pre**2)

        SL_input = tf.reshape(SL_input, [-1, 1, 1, 10])
        SL_target = tf.reshape(SL_target, [-1, 1, 1, 6])
        SL_weight = tf.reshape(SL_weight, [-1, 1, 1, 6])
        
        SL_pred = tf.reshape(self.actor('actor',SL_input, reuse=True), [-1, 1, 1, 6])
        SL_loss = tf.reduce_mean((SL_weight*(SL_pred-SL_target))**2)#SL_weight


        # Get variables
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "actor")
        critic_vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "critic1")
        
        #discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator")
        w_loss = tf.add_n([tf.nn.l2_loss(v) for v in actor_vars
                    if 'weight' in v.name]) 
        w_loss1 = tf.add_n([tf.nn.l2_loss(v) for v in critic_vars1
                    if 'weight' in v.name]) 
        
        critic_loss1 = TD_loss1 + 0.0001*w_loss1 
        pre_actor_loss = SL_loss + 0.0001*w_loss 
        actor_loss = -Q_loss 

        with tf.variable_scope('Adam_var_critic1'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "critic1") 
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(critic_learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(critic_loss1, var_list=critic_vars1))
                critic_opt1 = optimizer.apply_gradients(zip(gradients, variables))

        with tf.variable_scope('Adam_var_actor'):
            optimizer = tf.train.AdamOptimizer(actor_learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(actor_loss, var_list=actor_vars))
            actor_opt = optimizer.apply_gradients(zip(gradients, variables))

        with tf.variable_scope('Adam_var_actor_pre'):
            optimizer = tf.train.AdamOptimizer(actor_learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(pre_actor_loss, var_list=actor_vars))
            pre_actor_opt = optimizer.apply_gradients(zip(gradients, variables))


        return critic_loss1, TD_loss1, w_loss1, \
               actor_loss, SL_loss, critic_opt1, pre_actor_opt, actor_opt, action_MS


#Simulation parameter
pi = np.pi
xl = 4.0*pi; yl = 2.0; zl = 2.0*pi;
nxp = 48; nyp = 49; nzp = 48; nx = int(nxp/3*2); ny = nyp; nz = int(nzp/3*2);

dx = xl/nx; dz = zl/nz; 
Re = 180.0
dt = 0.0001
Cs = 0.08

nyp_rl = 1; 
nzp_rl = 8; nzp_r = nzp//nzp_rl; nxp_rl = 8; nxp_r = nxp//nxp_rl; 
input_std = np.ones([1,1,1,1,9], dtype=np.float64);

# n_sim : the number of simulatons
# nt : the number of time steps for one DRL step
n_sim = 8; nt = 30; 
dt_gpu = cp.asarray(np.zeros([n_sim,1,1,1,1], dtype=np.float64))
dt_gpu[:] = dt
t_s = np.zeros([n_sim], dtype=np.int32); t_e = np.zeros([n_sim], dtype=np.int32);
episode = np.zeros([n_sim], dtype=np.int32)
u = np.zeros([n_sim,nyp+1,nzp,nxp,2], dtype=np.float64) #horizontal u, w
v = np.zeros([n_sim,nyp,nzp,nxp,1], dtype=np.float64) #vertical v
up_mid = np.zeros([n_sim,nyp+1,nzp,nxp//2+1,2], dtype=np.complex128)
vp_mid = np.zeros([n_sim,nyp,nzp,nxp//2+1,1], dtype=np.complex128)

up_t = np.zeros([n_sim,nyp+1,nzp,nxp//2+1,2], dtype=np.complex128) #horizontal u, w
vp_t = np.zeros([n_sim,nyp,nzp,nxp//2+1,1], dtype=np.complex128) #vertical v
dudy_t = np.zeros([n_sim,nyp,nzp,nxp,2], dtype=np.float64)
dvdy_t = np.zeros([n_sim,nyp+1,nzp,nxp,1], dtype=np.float64)

up_t_gpu = cp.asarray(up_t)
vp_t_gpu = cp.asarray(vp_t)
dudy_t_gpu = cp.asarray(dudy_t)
dvdy_t_gpu = cp.asarray(dvdy_t)

n_initial = n_sim
buffer_up_ini = np.zeros([n_initial,nyp+1,nzp,nxp//2+1,2], dtype=np.complex128)
buffer_vp_ini = np.zeros([n_initial,nyp,nzp,nxp//2+1,1], dtype=np.complex128)
buffer_up_ini_gpu = cp.asarray(buffer_up_ini)
buffer_vp_ini_gpu = cp.asarray(buffer_vp_ini)

dudx = np.zeros([n_sim,nyp+1,nzp,nxp,2], dtype=np.float64)
dudy = np.zeros([n_sim,nyp,nzp,nxp,2], dtype=np.float64)
dudz = np.zeros([n_sim,nyp+1,nzp,nxp,2], dtype=np.float64)
H1p = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)
H2p = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)

dvdx = np.zeros([n_sim,nyp,nzp,nxp,1], dtype=np.float64)
dvdy = np.zeros([n_sim,nyp+1,nzp,nxp,1], dtype=np.float64)
dvdz = np.zeros([n_sim,nyp,nzp,nxp,1], dtype=np.float64)
Hv1p = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
Hv2p = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)

SGS1p = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)
SGS2p = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)

SGSv1p = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
SGSv2p = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)

nu_t1 = np.zeros([n_sim,nyp-1,nzp,nxp,1], dtype=np.float64)
nu_t2 = np.zeros([n_sim,nyp,nzp,nxp,1], dtype=np.float64)

pp = np.zeros([n_sim,nyp+1,nzp,nxp//2+1,1], dtype=np.complex128)
divp = np.zeros([n_sim,nyp+1,nzp,nxp//2+1,1], dtype=np.complex128)
y = np.zeros([1,nyp,1,1,1], dtype=np.float64)
dy = np.zeros([1,nyp+1,1,1,1], dtype=np.float64)
for j in range(nyp):#
    #y[0,j,:,:,:]  = -np.cos(float(j)/float(nyp-1)*pi)
    y[0,j,:,:,:] = -np.tanh(2.5*(1.0-2.*j/(nyp-1)))/np.tanh(2.5)
    #y[0,j,:,:,:] = -1 + 2*float(j)/float(nyp-1)

Filename4 = 'y_profile.plt'
fw = open(Filename4, 'w')
for j in range(nyp):
    fw.write('%f \n'%((1-np.abs(y[0,j,0,0,0]))*Re))
fw.close()

dy[0,1:nyp] = y[0,1:nyp] - y[0,0:nyp-1]; dy[0,0:1] = dy[0,1:2]; dy[0,nyp:nyp+1] = dy[0,nyp-1:nyp]

alpha = np.zeros([1,1,1,nxp//2+1,1], dtype=np.float64)
for k in range(nxp//2+1):
    alpha[0,0,0,k,0] = 2*pi/xl*k
gamma = np.zeros([1,1,nzp,1,1], dtype=np.float64)
for k in range(nzp//2):
    gamma[0,0,k,0,0] = 2*pi/zl*k
    gamma[0,0,nzp-k-1,0,0] = 2*pi/zl*(-k-1)

up_mid_gpu = cp.asarray(up_mid); vp_mid_gpu = cp.asarray(vp_mid);
y_gpu = cp.asarray(y); dy_gpu = cp.asarray(dy);
alpha_gpu = cp.asarray(alpha); gamma_gpu = cp.asarray(gamma);
dudx_gpu = cp.asarray(dudx); dudy_gpu = cp.asarray(dudy); dudz_gpu = cp.asarray(dudz);
dvdx_gpu = cp.asarray(dvdx); dvdy_gpu = cp.asarray(dvdy); dvdz_gpu = cp.asarray(dvdz);
H1p_gpu = cp.asarray(H1p); H2p_gpu = cp.asarray(H2p); 
Hv1p_gpu = cp.asarray(Hv1p); Hv2p_gpu = cp.asarray(Hv2p);

SGS1p_gpu = cp.asarray(SGS1p); SGS2p_gpu = cp.asarray(SGS2p);
SGSv1p_gpu = cp.asarray(SGSv1p); SGSv2p_gpu = cp.asarray(SGSv2p);

nu_t1_gpu = cp.asarray(nu_t1); nu_t2_gpu = cp.asarray(nu_t2);

pp_gpu = cp.asarray(pp)

a = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)
b = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)
c = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)
d = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)
a_gpu = cp.asarray(a)
b_gpu = cp.asarray(b)
c_gpu = cp.asarray(c)
d_gpu = cp.asarray(d)
wtdma = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,2], dtype=np.complex128)
gtdma = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)
ptdma = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,2], dtype=np.complex128)
wtdma_gpu = cp.asarray(wtdma)
gtdma_gpu = cp.asarray(gtdma)
ptdma_gpu = cp.asarray(ptdma)

av = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
bv = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
cv = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
dv = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
av_gpu = cp.asarray(av)
bv_gpu = cp.asarray(bv)
cv_gpu = cp.asarray(cv)
dv_gpu = cp.asarray(dv)
wtdma_v = np.zeros([n_sim,nyp-3,nzp,nxp//2+1,1], dtype=np.complex128)
gtdma_v = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
ptdma_v = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
wtdma_v_gpu = cp.asarray(wtdma_v)
gtdma_v_gpu = cp.asarray(gtdma_v)
ptdma_v_gpu = cp.asarray(ptdma_v)


app = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
bpp = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
cpp = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
dpp = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
ap_gpu = cp.asarray(app)
bp_gpu = cp.asarray(bpp)
cp_gpu = cp.asarray(cpp)
dp_gpu = cp.asarray(dpp)
wtdma_p = np.zeros([n_sim,nyp-2,nzp,nxp//2+1,1], dtype=np.complex128)
gtdma_p = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
ptdma_p = np.zeros([n_sim,nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
wtdma_p_gpu = cp.asarray(wtdma_p)
gtdma_p_gpu = cp.asarray(gtdma_p)
ptdma_p_gpu = cp.asarray(ptdma_p)


#pressure TDMA coefficient
ap_gpu[:,0:nyp-1] = 1.0/dy_gpu[:,1:nyp,:,:]/(0.5*dy_gpu[:,0:nyp-1]+0.5*dy_gpu[:,1:nyp])
bp_gpu[:,0:nyp-1] = -1.0*alpha_gpu**2-1.0*gamma_gpu**2-1.0/dy_gpu[:,1:nyp]*(1.0/(0.5*dy_gpu[:,0:nyp-1]+0.5*dy_gpu[:,1:nyp])
                                                                           +1.0/(0.5*dy_gpu[:,1:nyp]+0.5*dy_gpu[:,2:nyp+1]))
cp_gpu[:,0:nyp-1] = 1.0/dy_gpu[:,1:nyp,:,:]/(0.5*dy_gpu[:,1:nyp]+0.5*dy_gpu[:,2:nyp+1])

bp_gpu[:,0:1] = -1.0*alpha_gpu**2-1.0*gamma_gpu**2-1.0*(1.0/dy_gpu[:,1:2])/(0.5*dy_gpu[:,1:2]+0.5*dy_gpu[:,2:3])
bp_gpu[:,nyp-2:nyp-1] = -1.0*alpha_gpu**2-1.0*gamma_gpu**2-1.0*(1.0/dy_gpu[:,nyp-1:nyp])/(0.5*dy_gpu[:,nyp-2:nyp-1]+0.5*dy_gpu[:,nyp-1:nyp]) 

#To avoid singularity of TDMA matix
bp_gpu[:,0:1,0:1,0:1] = 1; cp_gpu[:,0:1,0:1,0:1] = 0; dp_gpu[:,0:1,0:1,0:1] = 0

def TDMA_gpu(a,b,c,d,w,g,p,n):

    w[:,0] = c[:,0]/b[:,0]
    g[:,0] = d[:,0]/b[:,0]

    for i in range(1,n-1):
        w[:,i] = c[:,i]/(b[:,i] - a[:,i]*w[:,i-1])
    for i in range(1,n):
        g[:,i] = (d[:,i] - a[:,i]*g[:,i-1])/(b[:,i] - a[:,i]*w[:,i-1])
    p[:,n-1] = g[:,n-1]
    for i in range(n-1,0,-1):
        p[:,i-1] = g[:,i-1] - w[:,i-1]*p[:,i]
    return p

def zero_padding(up): up[:,:,:,nx//2:,:] = 0; up[:,:,nz//2:nzp-(nz//2-1),:,:] = 0;
def filtering(up): up[:,:,:,nx//4:,:] = 0; up[:,:,nz//4:nzp-(nz//4-1),:,:] = 0;  
def test_filtering(up): up[:,:,:,nx//4:,:] = 0; up[:,:,nz//4:nzp-(nz//4-1),:,:] = 0;  

def random_phase_shift(batch,dz_shift):
    batch_p = np.fft.fft(batch, axis=1)
    phase = np.exp(-1j*dz_shift*np.reshape(gamma, (1,nzp,1,1)))
    batch_p = batch_p*phase

    batch[:] = np.real(np.fft.ifft(batch_p, axis=1))

def get_initial():
    #np.random.seed(0)
    u[:,1:nyp,:,:,0:2] = np.float64(0.1*np.random.normal(size=(n_sim,nyp-1,nzp,nxp,2)))
    v[:,1:nyp-1,:,:,0:1] = np.float64(0.1*np.random.normal(size=(n_sim,nyp-2,nzp,nxp,1)))

    # mean profile
    for j in range(1,nyp):
        y_vis = (1-(y[:,j-1]+0.5*dy[:,j]))*Re
        u_mean = np.mean(u[:,j:j+1,:,:,0:1])
        if y_vis < 15.0:
            u[:,j:j+1,:,:,0:1] = u[:,j:j+1,:,:,0:1]-u_mean + (y_vis)
        elif y_vis >= 15.0 and y_vis < Re:
            u[:,j:j+1,:,:,0:1] =  u[:,j:j+1,:,:,0:1]-u_mean + (1.0/0.41*np.log(y_vis) + 5.2)
        elif y_vis >= Re and 2*Re-y_vis > 15.0 :
            u[:,j:j+1,:,:,0:1] =  u[:,j:j+1,:,:,0:1]-u_mean + (1.0/0.41*np.log(2*Re-y_vis) + 5.2)
        elif y_vis >= Re and 2*Re-y_vis < 15.0 :
            u[:,j:j+1,:,:,0:1] =  u[:,j:j+1,:,:,0:1]-u_mean + (2*Re-y_vis)

    up = np.fft.rfft2(u, axes=(2,3))/(nzp*nxp)
    vp = np.fft.rfft2(v, axes=(2,3))/(nzp*nxp)
    zero_padding(up);zero_padding(vp);

    # large scale random fluctuations of initial condition
    up[:,1:nyp,0:1,0:1,1:2] = 0.0 #mean w
    up[:,1:nyp,1:5,1:5,0:2] = 0.5*up[:,1:nyp,1:5,1:5,0:2] / np.abs(up[:,1:nyp,1:5,1:5,0:2]) 
    up[:,1:nyp,nzp-1:nzp-5:-1,1:5,0:2] = 0.5*up[:,1:nyp,nzp-1:nzp-5:-1,1:5,0:2] / np.abs(up[:,1:nyp,nzp-1:nzp-5:-1,1:5,0:2])

    vp[:,1:nyp-1,0:1,0:1,0:1] = 0.0 #mean v
    vp[:,1:nyp-1,1:5,1:5,0:1] = 0.5*vp[:,1:nyp-1,1:5,1:5,0:1] / np.abs(vp[:,1:nyp-1,1:5,1:5,0:1]) 
    vp[:,1:nyp-1,nzp-1:nzp-5:-1,1:5,0:1] = 0.5*vp[:,1:nyp-1,nzp-1:nzp-5:-1,1:5,0:1] / np.abs(vp[:,1:nyp-1,nzp-1:nzp-5:-1,1:5,0:1])

    up_gpu = cp.asarray(up)
    vp_gpu = cp.asarray(vp)
    return up_gpu, vp_gpu


def time_loop(up_gpu,vp_gpu,SGS2p_gpu,SGSv2p_gpu,H2p_gpu,Hv2p_gpu,t_s,t_e, EQWM=False, SGSmodel=True, Random_action = False):

    for t in range(t_e[0]-t_s[0]):

        zero_padding(up_gpu); zero_padding(vp_gpu)
        up_gpu[:,0:1] = -up_gpu[:,1:2]; up_gpu[:,nyp:nyp+1] = -up_gpu[:,nyp-1:nyp];
        vp_gpu[:,0:1] = 0.0; vp_gpu[:,nyp-1:nyp] = 0.0; 
        #Get velocity and gradient of velocity (physical space)
        #Staggered grid (Horizontal velocity is on 0.5*dy, while vertical velocity is on dy)
        dudxp_gpu = 1j*alpha_gpu*up_gpu; dudzp_gpu = 1j*gamma_gpu*up_gpu
        dvdxp_gpu = 1j*alpha_gpu*vp_gpu; dvdzp_gpu = 1j*gamma_gpu*vp_gpu
        
        u_gpu = cp.fft.irfft2(up_gpu, axes=(2,3))*(nzp*nxp)
        v_gpu = cp.fft.irfft2(vp_gpu, axes=(2,3))*(nzp*nxp)
        
        dudx_gpu = cp.fft.irfft2(dudxp_gpu, axes=(2,3))*(nzp*nxp)
        dudz_gpu = cp.fft.irfft2(dudzp_gpu, axes=(2,3))*(nzp*nxp)
        dudy_gpu[:,0:nyp] = (u_gpu[:,1:nyp+1]-u_gpu[:,0:nyp])/(0.5*dy_gpu[:,0:nyp]+0.5*dy_gpu[:,1:nyp+1])  
     
        dvdx_gpu = cp.fft.irfft2(dvdxp_gpu, axes=(2,3))*(nzp*nxp)
        dvdz_gpu = cp.fft.irfft2(dvdzp_gpu, axes=(2,3))*(nzp*nxp)
        dvdy_gpu[:,1:nyp] = (v_gpu[:,1:nyp]-v_gpu[:,0:nyp-1])/(dy_gpu[:,1:nyp])
        dvdy_gpu[:,0:1] = -dvdy_gpu[:,1:2]; dvdy_gpu[:,nyp:nyp+1] = -dvdy_gpu[:,nyp-1:nyp];


        CFL_gpu = cp.amax(cp.abs(u_gpu[:,1:nyp,:,:,0:1])*dt_gpu/dx+
                          cp.abs((0.5*v_gpu[:,0:nyp-1]+0.5*v_gpu[:,1:nyp]))*dt_gpu/dy_gpu[:,1:nyp]+
                          cp.abs(u_gpu[:,1:nyp,:,:,1:2])*dt_gpu/dz, axis=(1,2,3,4), keepdims=True)
        dt_gpu[:] = dt_gpu[:]*0.15/CFL_gpu

        dt = cp.asnumpy(dt_gpu); CFL = cp.asnumpy(CFL_gpu);


        D_tensor_gpu = 1.0/Re*cp.concatenate([dudx_gpu[:,1:nyp,:,:,0:1], 
                              (0.5*dvdx_gpu[:,0:nyp-1]+0.5*dvdx_gpu[:,1:nyp]), 
                              dudx_gpu[:,1:nyp,:,:,1:2], 
                              (0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]+0.5*dudy_gpu[:,1:nyp,:,:,0:1]), 
                              dvdy_gpu[:,1:nyp], 
                              (0.5*dudy_gpu[:,0:nyp-1,:,:,1:2]+0.5*dudy_gpu[:,1:nyp,:,:,1:2]), 
                              dudz_gpu[:,1:nyp,:,:,0:1], 
                              (0.5*dvdz_gpu[:,0:nyp-1]+0.5*dvdz_gpu[:,1:nyp]), 
                              dudz_gpu[:,1:nyp,:,:,1:2]], axis=4) #0.5dy
        D_tensor = cp.asnumpy(D_tensor_gpu) / (input_std[:] + 10**-5)

        #dudx, (dudy+dvdx)/2, (dudz+dwdx)/2,
        #(dvdx+dudy)/2, dvdy, (dv/dz+dw/dy)/2
        #(dwdx+dudz)/2, (dwdy+dvdz)/2, dwdz,
        S11 = dudx_gpu[:,0:nyp+1,:,:,0:1] #0.5dy
        S22 = dvdy_gpu[:,0:nyp+1,:,:,0:1] #0.5dy
        S33 = dudz_gpu[:,0:nyp+1,:,:,1:2] #0.5dy
        S12 = 0.5*dudy_gpu[:,0:nyp,:,:,0:1] + 0.5*dvdx_gpu[:,0:nyp,:,:,0:1] #dy
        S13 = 0.5*dudz_gpu[:,0:nyp+1,:,:,0:1] + 0.5*dudx_gpu[:,0:nyp+1,:,:,1:2] #0.5dy
        S23 = 0.5*dvdz_gpu[:,0:nyp,:,:,0:1] + 0.5*dudy_gpu[:,0:nyp,:,:,1:2] #dy

        S_abs = (2*(S11[:,1:nyp]**2+S22[:,1:nyp]**2+S33[:,1:nyp]**2
                        +2*((0.5*S12[:,0:nyp-1]+0.5*S12[:,1:nyp])**2
                           + S13[:,1:nyp]**2
                           +(0.5*S23[:,0:nyp-1]+0.5*S23[:,1:nyp])**2)))**0.5

        if SGSmodel==True :

            AA1 = + dudx_gpu[:,1:nyp,:,:,0:1]**2 \
                  + (0.5*dvdx_gpu[:,0:nyp-1]**2+0.5*dvdx_gpu[:,1:nyp]**2) \
                  + dudx_gpu[:,1:nyp,:,:,1:2]**2 \
                  + (0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]**2+0.5*dudy_gpu[:,1:nyp,:,:,0:1]**2) \
                  + dvdy_gpu[:,1:nyp]**2 \
                  + (0.5*dudy_gpu[:,0:nyp-1,:,:,1:2]**2+0.5*dudy_gpu[:,1:nyp,:,:,1:2]**2) \
                  + dudz_gpu[:,1:nyp,:,:,0:1]**2 \
                  + (0.5*dvdz_gpu[:,0:nyp-1]**2+0.5*dvdz_gpu[:,1:nyp]**2) \
                  + dudz_gpu[:,1:nyp,:,:,1:2]**2 #0.5dy

            B11 = +dx**2*dudx_gpu[:,1:nyp,:,:,0:1]**2 \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]**2+0.5*dudy_gpu[:,1:nyp,:,:,0:1]**2) \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,0:1]**2
            B22 = +dx**2*(0.5*dvdx_gpu[:,0:nyp-1]**2+0.5*dvdx_gpu[:,1:nyp]**2) \
                  +dy_gpu[:,1:nyp]**2*dvdy_gpu[:,1:nyp]**2 \
                  +dz**2*(0.5*dvdz_gpu[:,0:nyp-1]**2+0.5*dvdz_gpu[:,1:nyp]**2)
            B33 = +dx**2*dudx_gpu[:,1:nyp,:,:,1:2]**2 \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,1:2]**2+0.5*dudy_gpu[:,1:nyp,:,:,1:2]**2) \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,1:2]**2
            B12 = +dx**2*dudx_gpu[:,1:nyp,:,:,0:1]*(0.5*dvdx_gpu[:,0:nyp-1]+0.5*dvdx_gpu[:,1:nyp]) \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]+0.5*dudy_gpu[:,1:nyp,:,:,0:1])*dvdy_gpu[:,1:nyp] \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,0:1]*(0.5*dvdz_gpu[:,0:nyp-1]+0.5*dvdz_gpu[:,1:nyp])
            B13 = +dx**2*dudx_gpu[:,1:nyp,:,:,0:1]*dudx_gpu[:,1:nyp,:,:,1:2] \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]*dudy_gpu[:,0:nyp-1,:,:,1:2]+0.5*dudy_gpu[:,1:nyp,:,:,0:1]*dudy_gpu[:,1:nyp,:,:,1:2]) \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,0:1]*dudz_gpu[:,1:nyp,:,:,1:2]
            B23 = +dx**2*dudx_gpu[:,1:nyp,:,:,1:2]*(0.5*dvdx_gpu[:,0:nyp-1]+0.5*dvdx_gpu[:,1:nyp]) \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,1:2]+0.5*dudy_gpu[:,1:nyp,:,:,1:2])*dvdy_gpu[:,1:nyp] \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,1:2]*(0.5*dvdz_gpu[:,0:nyp-1]+0.5*dvdz_gpu[:,1:nyp])

             
            BB1 = B11*B22-B12**2+B11*B33-B13**2+B22*B33-B23**2

            nu_t1_gpu[:,0:nyp-1] = 2.5*Cs**2*(BB1/AA1)**0.5
            nu_t2_gpu[:,1:nyp-1] = 0.5*(nu_t1_gpu[:,0:nyp-2]+nu_t1_gpu[:,1:nyp-1])
            

            TAU11 = cp.fft.rfft2(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S11[:,1:nyp], axes=(2,3))/(nxp*nzp) #0.5dy
            TAU22 = cp.fft.rfft2(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S22[:,1:nyp], axes=(2,3))/(nxp*nzp) #0.5dy
            TAU33 = cp.fft.rfft2(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S33[:,1:nyp], axes=(2,3))/(nxp*nzp) #0.5dy
            TAU12 = cp.fft.rfft2(-2*nu_t2_gpu[:,0:nyp  ,:,:,0:1]*S12[:,0:nyp], axes=(2,3))/(nxp*nzp) #1.0dy
            TAU13 = cp.fft.rfft2(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S13[:,1:nyp], axes=(2,3))/(nxp*nzp) #0.5dy
            TAU23 = cp.fft.rfft2(-2*nu_t2_gpu[:,0:nyp  ,:,:,0:1]*S23[:,0:nyp], axes=(2,3))/(nxp*nzp) #1.0dy   

            if t==0 : 
                
                TAU_vector = np.zeros([n_sim,nyp-1,nzp,nxp,6], dtype=np.float64) 
                TAU_vector11 = cp.asnumpy(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S11[:,1:nyp])
                TAU_vector22 = cp.asnumpy(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S22[:,1:nyp])
                TAU_vector33 = cp.asnumpy(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S33[:,1:nyp])
                TAU_vector12 = cp.asnumpy(-2*nu_t2_gpu[:,0:nyp,:,:,0:1]*S12[:,0:nyp])
                TAU_vector12 = 0.5*(TAU_vector12[:,0:nyp-1] + TAU_vector12[:,1:nyp])
                TAU_vector13 = cp.asnumpy(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S13[:,1:nyp])
                TAU_vector23 = cp.asnumpy(-2*nu_t2_gpu[:,0:nyp,:,:,0:1]*S23[:,0:nyp])
                TAU_vector23 = 0.5*(TAU_vector23[:,0:nyp-1] + TAU_vector23[:,1:nyp])

                TAU_vector[:,:,:,:,0:1] = TAU_vector11[:]
                TAU_vector[:,:,:,:,1:2] = TAU_vector22[:]
                TAU_vector[:,:,:,:,2:3] = TAU_vector33[:]
                TAU_vector[:,:,:,:,3:4] = TAU_vector12[:]
                TAU_vector[:,:,:,:,4:5] = TAU_vector13[:]
                TAU_vector[:,:,:,:,5:6] = TAU_vector23[:]
                TAU_vector_gpu = cp.asarray(TAU_vector)

        else :

            TAU_vector = np.zeros([n_sim,nyp-1,nzp,nxp,6], dtype=np.float64) 
            Filter_size_vector = np.zeros([n_sim,nyp-1,nzp,nxp,1], dtype=np.float64) 
            Filter_size_vector[:] = (Re**3*dx*dz*dy[:,1:nyp])**(2/3)

            if Random_action == True :

                TAU_vector[:,:(nyp-1)] = np.concatenate([
                                      np.reshape(sess.run(actor_behavior1, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[0:1,0:(nyp-1)],Filter_size_vector[0:1,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor_behavior2, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[1:2,0:(nyp-1)],Filter_size_vector[1:2,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor_behavior3, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[2:3,0:(nyp-1)],Filter_size_vector[2:3,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor_behavior4, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[3:4,0:(nyp-1)],Filter_size_vector[3:4,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor_behavior5, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[4:5,0:(nyp-1)],Filter_size_vector[4:5,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor_behavior6, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[5:6,0:(nyp-1)],Filter_size_vector[5:6,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor_behavior7, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[6:7,0:(nyp-1)],Filter_size_vector[6:7,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor_behavior8, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[7:8,0:(nyp-1)],Filter_size_vector[7:8,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6])], axis=0)

            else :

                TAU_vector[:,:(nyp-1)] = np.concatenate([
                                      np.reshape(sess.run(actor, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[0:1,0:(nyp-1)],Filter_size_vector[0:1,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[1:2,0:(nyp-1)],Filter_size_vector[1:2,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[2:3,0:(nyp-1)],Filter_size_vector[2:3,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[3:4,0:(nyp-1)],Filter_size_vector[3:4,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[4:5,0:(nyp-1)],Filter_size_vector[4:5,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[5:6,0:(nyp-1)],Filter_size_vector[5:6,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[6:7,0:(nyp-1)],Filter_size_vector[6:7,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6]),
                                      np.reshape(sess.run(actor, feed_dict = {obs_pre:  np.reshape(np.concatenate([D_tensor[7:8,0:(nyp-1)],Filter_size_vector[7:8,0:(nyp-1)]],axis=4), [-1, 1, 1, 1, 10])}), [1,(nyp-1),nzp,nxp,6])], axis=0)


            TAU_vector_gpu = cp.asarray(TAU_vector)
            TAU_vector12_gpu = cp.concatenate([-TAU_vector_gpu[:,0:1,:,:,3:4],TAU_vector_gpu[:,0:nyp-1,:,:,3:4],-TAU_vector_gpu[:,nyp-2:nyp-1,:,:,3:4]], axis=1)
            TAU_vector12_gpu = 0.5*(TAU_vector12_gpu[:,0:nyp]  + TAU_vector12_gpu[:,1:nyp+1])
            TAU_vector23_gpu = cp.concatenate([-TAU_vector_gpu[:,0:1,:,:,5:6],TAU_vector_gpu[:,0:nyp-1,:,:,5:6],-TAU_vector_gpu[:,nyp-2:nyp-1,:,:,5:6]], axis=1)
            TAU_vector23_gpu = 0.5*(TAU_vector23_gpu[:,0:nyp]  + TAU_vector23_gpu[:,1:nyp+1])
            TAU11 = cp.fft.rfft2(TAU_vector_gpu[:,0:nyp-1,:,:,0:1], axes=(2,3))/(nxp*nzp) #0.5dy
            TAU22 = cp.fft.rfft2(TAU_vector_gpu[:,0:nyp-1,:,:,1:2], axes=(2,3))/(nxp*nzp) #0.5dy
            TAU33 = cp.fft.rfft2(TAU_vector_gpu[:,0:nyp-1,:,:,2:3], axes=(2,3))/(nxp*nzp) #0.5dy
            TAU12 = cp.fft.rfft2(TAU_vector12_gpu[:,0:nyp], axes=(2,3))/(nxp*nzp) #1.0dy
            TAU13 = cp.fft.rfft2(TAU_vector_gpu[:,0:nyp-1,:,:,4:5], axes=(2,3))/(nxp*nzp) #0.5dy
            TAU23 = cp.fft.rfft2(TAU_vector23_gpu[:,0:nyp], axes=(2,3))/(nxp*nzp) #1.0dy        

        if t == 0:

            obs_pre_np = np.zeros([n_sim,nyp-1,nzp,nxp,10], dtype=np.float64)
            state_pre_np = np.zeros([n_sim,nyp-1,nzp,nxp,10], dtype=np.float64)
            action_np = np.zeros([n_sim,nyp-1,nzp,nxp,6], dtype=np.float64)

            obs_pre_np[:,:,:,:,0:9] = D_tensor[:] #/ (input_std[:] + 10**-5)
            obs_pre_np[:,:,:,:,9:10] = (Re**3*dx*dz*dy[:,1:nyp])**(2/3)
            state_pre_np[:,:,:,:,0:9] = D_tensor[:] #/ (input_std[:] + 10**-5)
            state_pre_np[:,:,:,:,9:10] = (Re**3*dx*dz*dy[:,1:nyp])**(2/3)
            action_np[:] = TAU_vector[:]#TAU[:]#Cs_square[:]

            u_mean_gpu = cp.mean(u_gpu, axis=(2,3))
            v_mean_gpu = cp.mean(v_gpu, axis=(2,3))
            u_rms_gpu = cp.std(u_gpu, axis=(2,3))
            v_rms_gpu = cp.std(v_gpu, axis=(2,3))
            uv_mean_gpu = cp.mean((u_gpu[:,1:nyp,:,:,0:1]-cp.reshape(u_mean_gpu[:,1:nyp,0:1], [n_sim, nyp-1, 1, 1, 1]))*(0.5*v_gpu[:,0:nyp-1]+0.5*v_gpu[:,1:nyp]), axis=(2,3))
            dudy_mean_gpu = cp.mean(0.5*(dudy_gpu[:,0:nyp-1,:,:,0:2] + dudy_gpu[:,1:nyp,:,:,0:2]), axis=(2,3)) / Re
            nu_t_dudy_mean = cp.mean(-TAU_vector_gpu[:,:,:,:,3:4], axis=(2,3))


            u_mean_pre = cp.asnumpy(cp.concatenate([u_mean_gpu[:,1:nyp,0:1],0.5*v_mean_gpu[:,0:nyp-1]+0.5*v_mean_gpu[:,1:nyp],u_mean_gpu[:,1:nyp,1:2]], axis=2))
            u_rms_pre = cp.asnumpy(cp.concatenate([u_rms_gpu[:,1:nyp,0:1],0.5*v_rms_gpu[:,0:nyp-1]+0.5*v_rms_gpu[:,1:nyp],u_rms_gpu[:,1:nyp,1:2]], axis=2))
            uv_mean_pre = cp.asnumpy(uv_mean_gpu)
            dudy_mean_pre = cp.asnumpy(dudy_mean_gpu)
            nu_t_dudy_mean_pre = cp.asnumpy(nu_t_dudy_mean)


            SL_input_np = np.zeros([n_sim,nyp-1,nzp,nxp,10], dtype=np.float64)
            SL_target_np = np.zeros([n_sim,nyp-1,nzp,nxp,6], dtype=np.float64)

            SL_input_np[:,:,:,:,0:9] = D_tensor[:]
            SL_input_np[:,:,:,:,9:10] = (Re**3*dx*dz*dy[:,1:nyp])**(2/3)
            #SL_target_np[:] = D_tensor[:]

            #nu_t1 = sess.run(actor_behavior1, feed_dict = {obs_pre:  D_tensor[0:1]})
            AA1 = + dudx_gpu[:,1:nyp,:,:,0:1]**2 \
                  + (0.5*dvdx_gpu[:,0:nyp-1]**2+0.5*dvdx_gpu[:,1:nyp]**2) \
                  + dudx_gpu[:,1:nyp,:,:,1:2]**2 \
                  + (0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]**2+0.5*dudy_gpu[:,1:nyp,:,:,0:1]**2) \
                  + dvdy_gpu[:,1:nyp]**2 \
                  + (0.5*dudy_gpu[:,0:nyp-1,:,:,1:2]**2+0.5*dudy_gpu[:,1:nyp,:,:,1:2]**2) \
                  + dudz_gpu[:,1:nyp,:,:,0:1]**2 \
                  + (0.5*dvdz_gpu[:,0:nyp-1]**2+0.5*dvdz_gpu[:,1:nyp]**2) \
                  + dudz_gpu[:,1:nyp,:,:,1:2]**2 #0.5dy

            B11 = +dx**2*dudx_gpu[:,1:nyp,:,:,0:1]**2 \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]**2+0.5*dudy_gpu[:,1:nyp,:,:,0:1]**2) \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,0:1]**2
            B22 = +dx**2*(0.5*dvdx_gpu[:,0:nyp-1]**2+0.5*dvdx_gpu[:,1:nyp]**2) \
                  +dy_gpu[:,1:nyp]**2*dvdy_gpu[:,1:nyp]**2 \
                  +dz**2*(0.5*dvdz_gpu[:,0:nyp-1]**2+0.5*dvdz_gpu[:,1:nyp]**2)
            B33 = +dx**2*dudx_gpu[:,1:nyp,:,:,1:2]**2 \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,1:2]**2+0.5*dudy_gpu[:,1:nyp,:,:,1:2]**2) \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,1:2]**2
            B12 = +dx**2*dudx_gpu[:,1:nyp,:,:,0:1]*(0.5*dvdx_gpu[:,0:nyp-1]+0.5*dvdx_gpu[:,1:nyp]) \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]+0.5*dudy_gpu[:,1:nyp,:,:,0:1])*dvdy_gpu[:,1:nyp] \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,0:1]*(0.5*dvdz_gpu[:,0:nyp-1]+0.5*dvdz_gpu[:,1:nyp])
            B13 = +dx**2*dudx_gpu[:,1:nyp,:,:,0:1]*dudx_gpu[:,1:nyp,:,:,1:2] \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,0:1]*dudy_gpu[:,0:nyp-1,:,:,1:2]+0.5*dudy_gpu[:,1:nyp,:,:,0:1]*dudy_gpu[:,1:nyp,:,:,1:2]) \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,0:1]*dudz_gpu[:,1:nyp,:,:,1:2]
            B23 = +dx**2*dudx_gpu[:,1:nyp,:,:,1:2]*(0.5*dvdx_gpu[:,0:nyp-1]+0.5*dvdx_gpu[:,1:nyp]) \
                  +dy_gpu[:,1:nyp]**2*(0.5*dudy_gpu[:,0:nyp-1,:,:,1:2]+0.5*dudy_gpu[:,1:nyp,:,:,1:2])*dvdy_gpu[:,1:nyp] \
                  +dz**2*dudz_gpu[:,1:nyp,:,:,1:2]*(0.5*dvdz_gpu[:,0:nyp-1]+0.5*dvdz_gpu[:,1:nyp])

             
            BB1 = B11*B22-B12**2+B11*B33-B13**2+B22*B33-B23**2

            nu_t1_gpu[:,0:nyp-1] = 2.5*Cs**2*(BB1/AA1)**0.5
            nu_t2_gpu[:,1:nyp-1] = 0.5*(nu_t1_gpu[:,0:nyp-2]+nu_t1_gpu[:,1:nyp-1])#2.5*Cs**2*(BB2[:,1:nyp-1]/AA2[:,1:nyp-1])**0.5   
              
            TAU_vector11 = cp.asnumpy(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S11[:,1:nyp])
            TAU_vector22 = cp.asnumpy(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S22[:,1:nyp])
            TAU_vector33 = cp.asnumpy(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S33[:,1:nyp])
            TAU_vector12 = cp.asnumpy(-2*nu_t2_gpu[:,0:nyp,:,:,0:1]*S12[:,0:nyp])
            TAU_vector12 = 0.5*(TAU_vector12[:,0:nyp-1] + TAU_vector12[:,1:nyp])
            TAU_vector13 = cp.asnumpy(-2*nu_t1_gpu[:,0:nyp-1,:,:,0:1]*S13[:,1:nyp])
            TAU_vector23 = cp.asnumpy(-2*nu_t2_gpu[:,0:nyp,:,:,0:1]*S23[:,0:nyp])
            TAU_vector23 = 0.5*(TAU_vector23[:,0:nyp-1] + TAU_vector23[:,1:nyp])

            TAU_vector[:,:,:,:,0:1] = TAU_vector11[:]
            TAU_vector[:,:,:,:,1:2] = TAU_vector22[:]
            TAU_vector[:,:,:,:,2:3] = TAU_vector33[:]
            TAU_vector[:,:,:,:,3:4] = TAU_vector12[:]
            TAU_vector[:,:,:,:,4:5] = TAU_vector13[:]
            TAU_vector[:,:,:,:,5:6] = TAU_vector23[:]

            SL_target_np[:] = TAU_vector[:,:,:,:,:] 

  
        SGS2_gpu = cp.concatenate((1j*alpha_gpu*TAU11[:]+(TAU12[:,1:nyp]-TAU12[:,0:nyp-1])/dy_gpu[:,1:nyp]+1j*gamma_gpu*TAU13[:],
                                   1j*alpha_gpu*TAU13[:]+(TAU23[:,1:nyp]-TAU23[:,0:nyp-1])/dy_gpu[:,1:nyp]+1j*gamma_gpu*TAU33[:])
                                   ,axis=4)
        SGSv2_gpu = +1j*alpha_gpu*TAU12[:,1:nyp-1]\
                    +(TAU22[:,1:nyp-1]-TAU22[:,0:nyp-2])/(0.5*dy_gpu[:,1:nyp-1]+0.5*dy_gpu[:,2:nyp])\
                    +1j*gamma_gpu*TAU23[:,1:nyp-1]

        #SGS term momentum (u,w)
        SGS1p_gpu[:] = SGS2p_gpu[:]
        SGS2p_gpu[:] = SGS2_gpu[:]#cp.fft.rfft2(SGS2_gpu, axes=(1,2))/(nxp*nzp)


        #SGS term momentum (v)
        SGSv1p_gpu[:] = SGSv2p_gpu[:]
        SGSv2p_gpu[:] = SGSv2_gpu[:]#cp.fft.rfft2(SGSv2_gpu, axes=(1,2))/(nxp*nzp)



        #nonlinear term momentum
        H1p_gpu[:] = H2p_gpu[:]
        H2_gpu = +u_gpu[:,1:nyp,:,:,0:1]*dudx_gpu[:,1:nyp]\
                 +0.5*(v_gpu[:,0:nyp-1,:,:,0:1]*dudy_gpu[:,0:nyp-1])+0.5*(v_gpu[:,1:nyp,:,:,0:1]*dudy_gpu[:,1:nyp])\
                 +u_gpu[:,1:nyp,:,:,1:2]*dudz_gpu[:,1:nyp]

        H2_gpu[:,:,:,:,0:1] += -1 #<dp/dx>=-1
        H2p_gpu[:] = cp.fft.rfft2(H2_gpu, axes=(2,3))/(nxp*nzp)


        #nonlinear term momentum
        Hv1p_gpu[:] = Hv2p_gpu[:]
        Hv2_gpu = +(0.5*u_gpu[:,1:nyp-1,:,:,0:1]+0.5*u_gpu[:,2:nyp,:,:,0:1])*dvdx_gpu[:,1:nyp-1]\
                  +v_gpu[:,1:nyp-1,:,:,0:1]*(0.5*dvdy_gpu[:,1:nyp-1]+0.5*dvdy_gpu[:,2:nyp])\
                  +(0.5*u_gpu[:,1:nyp-1,:,:,1:2]+0.5*u_gpu[:,2:nyp,:,:,1:2])*dvdz_gpu[:,1:nyp-1]
        Hv2p_gpu[:] = cp.fft.rfft2(Hv2_gpu, axes=(2,3))/(nxp*nzp)

        #first step --> euler

        if t==0 :
            for n_sim_idx in range(n_sim):
                if t_s[n_sim_idx] == 0: 
                    SGS1p_gpu[n_sim_idx:n_sim_idx+1] = SGS2p_gpu[n_sim_idx:n_sim_idx+1]
                    SGSv1p_gpu[n_sim_idx:n_sim_idx+1] = SGSv2p_gpu[n_sim_idx:n_sim_idx+1]
                    H1p_gpu[n_sim_idx:n_sim_idx+1] = H2p_gpu[n_sim_idx:n_sim_idx+1]
                    Hv1p_gpu[n_sim_idx:n_sim_idx+1] = Hv2p_gpu[n_sim_idx:n_sim_idx+1]


        #velocity TDMA
        d_gpu = ((1.0-dt_gpu/(2*Re)*alpha_gpu**2-dt_gpu/(2*Re)*gamma_gpu**2)*up_gpu[:,1:nyp]
                 +dt_gpu/(2*Re)*(+(up_gpu[:,2:nyp+1]-up_gpu[:,1:nyp])/(0.5*dy_gpu[:,1:nyp]+0.5*dy_gpu[:,2:nyp+1])
                             -(up_gpu[:,1:nyp]-up_gpu[:,0:nyp-1])/(0.5*dy_gpu[:,0:nyp-1]+0.5*dy_gpu[:,1:nyp]))
                             /(dy_gpu[:,1:nyp])
                 +dt_gpu*(-1.5*H2p_gpu[:,0:nyp-1]+0.5*H1p_gpu[:,0:nyp-1])
                 +dt_gpu*(-1.5*SGS2p_gpu[:,0:nyp-1]+0.5*SGS1p_gpu[:,0:nyp-1]))

        #vertical velocity TDMA
        dv_gpu = ((1.0-dt_gpu/(2*Re)*alpha_gpu**2-dt_gpu/(2*Re)*gamma_gpu**2)*vp_gpu[:,1:nyp-1]
                 +dt_gpu/(2*Re)*(+(vp_gpu[:,2:nyp]-vp_gpu[:,1:nyp-1])/dy_gpu[:,2:nyp]
                             -(vp_gpu[:,1:nyp-1]-vp_gpu[:,0:nyp-2])/dy_gpu[:,1:nyp-1])
                             /(0.5*dy_gpu[:,1:nyp-1]+0.5*dy_gpu[:,2:nyp])
                 +dt_gpu*(-1.5*Hv2p_gpu[:,0:nyp-2]+0.5*Hv1p_gpu[:,0:nyp-2])
                 +dt_gpu*(-1.5*SGSv2p_gpu[:,0:nyp-2]+0.5*SGSv1p_gpu[:,0:nyp-2]))


        #velocity u,w TDMA coefficient
        a_gpu[:,0:nyp-1] = -dt_gpu/(2*Re)/dy_gpu[:,1:nyp]/(0.5*dy_gpu[:,0:nyp-1]+0.5*dy_gpu[:,1:nyp])
        b_gpu[:,0:nyp-1] = 1.0+dt_gpu/(2*Re)*alpha_gpu**2+dt_gpu/(2*Re)*gamma_gpu**2+dt_gpu/(2*Re)/dy_gpu[:,1:nyp]*(1.0/(0.5*dy_gpu[:,0:nyp-1]+0.5*dy_gpu[:,1:nyp])
                                                                                                       +1.0/(0.5*dy_gpu[:,1:nyp]+0.5*dy_gpu[:,2:nyp+1]))
        c_gpu[:,0:nyp-1] = -dt_gpu/(2*Re)/dy_gpu[:,1:nyp]/(0.5*dy_gpu[:,1:nyp]+0.5*dy_gpu[:,2:nyp+1])

        b_gpu[:,0:1] = 1.0+dt_gpu/(2*Re)*alpha_gpu**2+dt_gpu/(2*Re)*gamma_gpu**2+dt_gpu/(2*Re)/dy_gpu[:,1:2]*(2.0/(0.5*dy_gpu[:,0:1]+0.5*dy_gpu[:,1:2])
                                                                                                       +1.0/(0.5*dy_gpu[:,1:2]+0.5*dy_gpu[:,2:3]))

        b_gpu[:,nyp-2:nyp-1] = 1.0+dt_gpu/(2*Re)*alpha_gpu**2+dt_gpu/(2*Re)*gamma_gpu**2+dt_gpu/(2*Re)/dy_gpu[:,nyp-1:nyp]*(1.0/(0.5*dy_gpu[:,nyp-2:nyp-1]+0.5*dy_gpu[:,nyp-1:nyp])
                                                                                                       +2.0/(0.5*dy_gpu[:,nyp-1:nyp]+0.5*dy_gpu[:,nyp:nyp+1]))
        #velocity v TDMA coefficient
        av_gpu[:,0:nyp-2] = -dt_gpu/(2*Re)/dy_gpu[:,1:nyp-1]/(0.5*dy_gpu[:,1:nyp-1]+0.5*dy_gpu[:,2:nyp])
        bv_gpu[:,0:nyp-2] = 1.0+dt_gpu/(2*Re)*alpha_gpu**2+dt_gpu/(2*Re)*gamma_gpu**2+dt_gpu/(2*Re)*(1.0/dy_gpu[:,1:nyp-1]+1.0/dy_gpu[:,2:nyp])/(0.5*dy_gpu[:,1:nyp-1]+0.5*dy_gpu[:,2:nyp])
        cv_gpu[:,0:nyp-2] = -dt_gpu/(2*Re)/dy_gpu[:,2:nyp]/(0.5*dy_gpu[:,1:nyp-1]+0.5*dy_gpu[:,2:nyp])


        up_mid_gpu[:,1:nyp,:,:,:] = TDMA_gpu(a_gpu,b_gpu,c_gpu,d_gpu,wtdma_gpu,gtdma_gpu,ptdma_gpu,nyp-1)


        vp_mid_gpu[:,1:nyp-1,:,:,:] = TDMA_gpu(av_gpu,bv_gpu,cv_gpu,dv_gpu,wtdma_v_gpu,gtdma_v_gpu,ptdma_v_gpu,nyp-2)        
        vp_mid_gpu[:,0:1] = 0.0
        vp_mid_gpu[:,nyp-1:nyp] = 0.0


        #pressure TDMA
        dp_gpu[:,0:nyp-1,:,:] = 1.0/dt_gpu*(+1j*alpha_gpu*(up_mid_gpu[:,1:nyp,:,:,0:1]) 
                                        +1j*gamma_gpu*(up_mid_gpu[:,1:nyp,:,:,1:2])
                                        +(vp_mid_gpu[:,1:nyp,:,:,0:1]-vp_mid_gpu[:,0:nyp-1,:,:,0:1])/dy_gpu[:,1:nyp])


        pp_gpu[:,1:nyp,:,:,:] = TDMA_gpu(ap_gpu,bp_gpu,cp_gpu,dp_gpu,wtdma_p_gpu,gtdma_p_gpu,ptdma_p_gpu,nyp-1)
        
        pp_gpu[:,0:1,:,:] = pp_gpu[:,1:2,:,:]
        pp_gpu[:,nyp:nyp+1,:,:] = pp_gpu[:,nyp-1:nyp,:,:]
           
        #get pressure gradient
        dpdxp_gpu = 1j*alpha_gpu*(pp_gpu[:,0:nyp+1])
        dpdzp_gpu = 1j*gamma_gpu*(pp_gpu[:,0:nyp+1])                     
        dpdyp_gpu = (pp_gpu[:,1:nyp+1]-pp_gpu[:,0:nyp])/(0.5*dy_gpu[:,1:nyp+1]+0.5*dy_gpu[:,0:nyp])

        ##mean pressure gradient (<dpdx> = -1)
        # pressure correction
        up_gpu[:,1:nyp,:,:,0:1] = up_mid_gpu[:,1:nyp,:,:,0:1] - dt_gpu * dpdxp_gpu[:,1:nyp]
        up_gpu[:,1:nyp,:,:,1:2] = up_mid_gpu[:,1:nyp,:,:,1:2] - dt_gpu * dpdzp_gpu[:,1:nyp]
        vp_gpu[:,0:nyp,:,:,0:1] = vp_mid_gpu[:,0:nyp,:,:,0:1] - dt_gpu * dpdyp_gpu[:,0:nyp]


        if t == t_e[0]-t_s[0]-1:

            zero_padding(up_gpu); zero_padding(vp_gpu)
            up_gpu[:,0:1] = -up_gpu[:,1:2]; up_gpu[:,nyp:nyp+1] = -up_gpu[:,nyp-1:nyp];
            #vp_gpu[:,0:1] = 0.0; vp_gpu[:,nyp-1:nyp] = 0.0; 

            dudxp_gpu = 1j*alpha_gpu*up_gpu; dudzp_gpu = 1j*gamma_gpu*up_gpu
            dvdxp_gpu = 1j*alpha_gpu*vp_gpu; dvdzp_gpu = 1j*gamma_gpu*vp_gpu
            
            u_gpu = cp.fft.irfft2(up_gpu, axes=(2,3))*(nzp*nxp)
            v_gpu = cp.fft.irfft2(vp_gpu, axes=(2,3))*(nzp*nxp)
            
            dudx_gpu = cp.fft.irfft2(dudxp_gpu, axes=(2,3))*(nzp*nxp)
            dudz_gpu = cp.fft.irfft2(dudzp_gpu, axes=(2,3))*(nzp*nxp)
            dudy_gpu[:,0:nyp] = (u_gpu[:,1:nyp+1]-u_gpu[:,0:nyp])/(0.5*dy_gpu[:,0:nyp]+0.5*dy_gpu[:,1:nyp+1])  
         
            dvdx_gpu = cp.fft.irfft2(dvdxp_gpu, axes=(2,3))*(nzp*nxp)
            dvdz_gpu = cp.fft.irfft2(dvdzp_gpu, axes=(2,3))*(nzp*nxp)
            dvdy_gpu[:,1:nyp] = (v_gpu[:,1:nyp]-v_gpu[:,0:nyp-1])/(dy_gpu[:,1:nyp])
            dvdy_gpu[:,0:1] = -dvdy_gpu[:,1:2]; dvdy_gpu[:,nyp:nyp+1] = -dvdy_gpu[:,nyp-1:nyp];

            u_mean_gpu = cp.mean(u_gpu, axis=(2,3))
            v_mean_gpu = cp.mean(v_gpu, axis=(2,3))
            u_rms_gpu = cp.std(u_gpu, axis=(2,3))
            v_rms_gpu = cp.std(v_gpu, axis=(2,3))
            uv_mean_gpu = cp.mean((u_gpu[:,1:nyp,:,:,0:1]-cp.reshape(u_mean_gpu[:,1:nyp,0:1], [n_sim, nyp-1, 1, 1, 1]))*(0.5*v_gpu[:,0:nyp-1]+0.5*v_gpu[:,1:nyp]), axis=(2,3))
            dudy_mean_gpu = cp.mean(0.5*(dudy_gpu[:,0:nyp-1,:,:,0:2] + dudy_gpu[:,1:nyp,:,:,0:2]), axis=(2,3)) / Re
            nu_t_dudy_mean = cp.mean(-TAU_vector_gpu[:,:,:,:,3:4], axis=(2,3))

            u_mean_next = cp.asnumpy(cp.concatenate([u_mean_gpu[:,1:nyp,0:1],0.5*v_mean_gpu[:,0:nyp-1]+0.5*v_mean_gpu[:,1:nyp],u_mean_gpu[:,1:nyp,1:2]], axis=2))
            u_rms_next = cp.asnumpy(cp.concatenate([u_rms_gpu[:,1:nyp,0:1],0.5*v_rms_gpu[:,0:nyp-1]+0.5*v_rms_gpu[:,1:nyp],u_rms_gpu[:,1:nyp,1:2]], axis=2))
            dudy_mean_next = cp.asnumpy(dudy_mean_gpu)
            uv_mean_next = cp.asnumpy(uv_mean_gpu)
            nu_t_dudy_mean_next = cp.asnumpy(nu_t_dudy_mean)

    return state_pre_np, obs_pre_np, action_np, \
           u_mean_pre, u_rms_pre, uv_mean_pre, dudy_mean_pre, u_mean_next, u_rms_next, uv_mean_next, dudy_mean_next, nu_t_dudy_mean_pre, nu_t_dudy_mean_next, \
           SL_input_np, SL_target_np


batch_size = 64
batch_state_pre = np.zeros([batch_size, 1, nzp_rl, nxp_rl, 10], dtype=np.float32)
batch_state_stat_pre = np.zeros([batch_size, 1, nzp_rl, nxp_rl, 9], dtype=np.float32)
batch_obs_pre = np.zeros([batch_size, 1, nzp_rl, nxp_rl, 10], dtype=np.float32)
batch_action = np.zeros([batch_size, 1, nzp_rl, nxp_rl, 6], dtype=np.float32)
batch_reward = np.zeros([batch_size, 1, 1, 1, 1], dtype=np.float32)
batch_done = np.zeros([batch_size, 1], dtype=np.float32)

batch_state_action_mean = np.zeros([batch_size, 16+9], dtype=np.float32)
batch_state_action_std = np.zeros([batch_size, 16+9], dtype=np.float32)

batch_size_SL = 64
batch_SL_input = np.zeros([batch_size_SL, 1, 1, 1, 10], dtype=np.float32)
batch_SL_target = np.zeros([batch_size_SL, 1, 1, 1, 6], dtype=np.float32)
batch_SL_weight = np.zeros([batch_size_SL, 1, 1, 1, 6], dtype=np.float32)

#--------------------------- Model
model = DRL()
data_shape = batch_state_pre.shape

state_pre, state_stat_pre, obs_pre, action, reward, state_action_mean, state_action_std, \
critic_learning_rate, actor_learning_rate, \
SL_input, SL_target, SL_weight\
= model.model_inputs(data_shape[1], data_shape[2], data_shape[3])

critic_loss1, TD_loss1, w_loss1, \
actor_loss, SL_loss, critic_opt1, pre_actor_opt, actor_opt, action_MS \
= model.model_train(state_pre, state_stat_pre, obs_pre, action, reward, 
                    critic_learning_rate, actor_learning_rate, SL_input, SL_target, SL_weight)

actor = model.actor('actor',obs_pre, reuse=True)
actor_behavior1 = model.actor('actor_behavior1',obs_pre, reuse=False)
actor_behavior2 = model.actor('actor_behavior2',obs_pre, reuse=False)
actor_behavior3 = model.actor('actor_behavior3',obs_pre, reuse=False)
actor_behavior4 = model.actor('actor_behavior4',obs_pre, reuse=False)
actor_behavior5 = model.actor('actor_behavior5',obs_pre, reuse=False)
actor_behavior6 = model.actor('actor_behavior6',obs_pre, reuse=False)
actor_behavior7 = model.actor('actor_behavior7',obs_pre, reuse=False)
actor_behavior8 = model.actor('actor_behavior8',obs_pre, reuse=False)

critic1 = model.critic('critic1',tf.concat([state_pre,action], axis=4), state_stat_pre, state_action_mean, state_action_std, reuse=True)
critic_actor1 = model.critic('critic1',tf.concat([state_pre,tf.reshape(actor,[-1,1,nzp_rl,nxp_rl,6])], axis=4), state_stat_pre, state_action_mean, state_action_std, reuse=True)

#---------------------------
#---------------------------
def get_noise_var():  
    vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'actor')
    noise_vars = [tf.Variable(tf.zeros(var.shape),dtype=tf.float32) for var in vars1]
    return noise_vars 

tf_ada_std = tf.placeholder(tf.float32, [], name='ada_std')

noise_vars1 = get_noise_var()
noise_vars2 = get_noise_var()
noise_vars3 = get_noise_var()
noise_vars4 = get_noise_var()
noise_vars5 = get_noise_var()
noise_vars6 = get_noise_var()
noise_vars7 = get_noise_var()
noise_vars8 = get_noise_var()

#---------------------------
saver = tf.train.Saver(max_to_keep=None)
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

#saver.restore(sess, './24000/MODEL.ckpt-24000')
#print('Restore success')


def OUNoise(X, noise_std, mu=0, std=0.01):
    return X + 0.1*(0.05*(mu - X) + noise_std*np.random.randn(*(X.shape)))

def vars_assign(NET1,NET2):
    var_names = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, NET1)
    var_names2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, NET2)
    old_var = sess.run(var_names)
    var_shapes = [i.shape for i in old_var]
    var_placeholder = [tf.placeholder(tf.float32, j) for i,j in zip(old_var,var_shapes)]
    assign_op = [i.assign(j) for i,j in zip(var_names2,var_placeholder)]

    return assign_op, var_placeholder

def noise_update(NET2,noise_vars,ada_std=0.5):
    vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'actor')
    vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, NET2)   

    op_holder1 = []
    for var1, noise_var in zip(vars1,noise_vars):
        op_holder1.append(noise_var.assign(noise_var + (0.001*(0 - noise_var) + tf_ada_std*tf.math.reduce_std(var1)*tf.random_normal(noise_var.shape))))   

    op_holder2 = []
    for from_var,noise_var,to_var in zip(vars1,noise_vars,vars2):
        op_holder2.append(to_var.assign(from_var+noise_var))    

     
    return op_holder1, op_holder2

ada_std = np.zeros([n_sim], dtype=np.float64)
ada_std_in = np.ones([n_sim], dtype=np.float64)
ada_std[:] = 0.01
noise_update_op1, noise_update_actor_op1 = noise_update('actor_behavior1',noise_vars1,ada_std)
noise_update_op2, noise_update_actor_op2 = noise_update('actor_behavior2',noise_vars2,ada_std)
noise_update_op3, noise_update_actor_op3 = noise_update('actor_behavior3',noise_vars3,ada_std)
noise_update_op4, noise_update_actor_op4 = noise_update('actor_behavior4',noise_vars4,ada_std)
noise_update_op5, noise_update_actor_op5 = noise_update('actor_behavior5',noise_vars5,ada_std)
noise_update_op6, noise_update_actor_op6 = noise_update('actor_behavior6',noise_vars6,ada_std)
noise_update_op7, noise_update_actor_op7 = noise_update('actor_behavior7',noise_vars7,ada_std)
noise_update_op8, noise_update_actor_op8 = noise_update('actor_behavior8',noise_vars8,ada_std)


stat_DNS = np.zeros([1,(129-1),8], dtype=np.float64) 
y_DNS = np.zeros([129,1], dtype=np.float64)
f = h5py.File('DNS_Re180_statistics_32x32.hdf5', 'r')
y_DNS[:,0] = f["y"].value[:]# - 1 #y/delta
stat_DNS[:,:,0:1] = f["u_mean"].value[:,:,0:1]
stat_DNS[:,:,3:6] = f["u_rms"].value[:]
stat_DNS[:,:,6:7] = f["dudy_mean"].value[:,:,0:1]
stat_DNS[:,:,7:8] = f["uv_mean"].value[:]


stat_target = np.zeros([1,(nyp-1),8], dtype=np.float64) 
f = interp1d(0.5*(y_DNS[:128,0]+y_DNS[1:129,0]), stat_DNS[0,:,:], axis=0)
stat_target[0,:,:] = f(y[0,0:(nyp-1),0,0,0]+0.5*dy[0,1:nyp,0,0,0])

Filename4 = 'Target_statistics.plt'
fw = open(Filename4, 'w')
fw.write('VARIABLES="y","y+","u_mean","v_mean","w_mean","u_rms","v_rms","w_rms","dudy","uv"\n') 
fw.write('Zone T="wall"\n')
for j in range((nyp-1)):
    fw.write('%f %f %f %f %f %f %f %f %f %f \n'%(y[0,j,0]+0.5*dy[0,j+1],(y[0,j,0]+0.5*dy[0,j+1]+1)*Re,
                                           stat_target[0,j,0],stat_target[0,j,1],stat_target[0,j,2],
                                           stat_target[0,j,3],stat_target[0,j,4],stat_target[0,j,5],
                                           stat_target[0,j,6],stat_target[0,j,7]))
fw.close()


tavg_stat_mean = np.zeros([n_sim,(nyp-1),3], dtype=np.float64) 
tavg_stat_rms = np.zeros([n_sim,(nyp-1),3], dtype=np.float64) 
tavg_stat_uv = np.zeros([n_sim,(nyp-1),1], dtype=np.float64) 
tavg_stat_dudy = np.zeros([n_sim,(nyp-1),2], dtype=np.float64) 
tavg_stat_nu_t_dudy = np.zeros([n_sim,(nyp-1),1], dtype=np.float64) 
tavg_stat_mean_next = np.zeros([n_sim,(nyp-1),3], dtype=np.float64) 
tavg_stat_rms_next = np.zeros([n_sim,(nyp-1),3], dtype=np.float64) 
tavg_stat_uv_next = np.zeros([n_sim,(nyp-1),1], dtype=np.float64) 
tavg_stat_dudy_next = np.zeros([n_sim,(nyp-1),2], dtype=np.float64) 
tavg_stat_nu_t_dudy_next = np.zeros([n_sim,(nyp-1),1], dtype=np.float64) 


print('ready for buffer')
ini_buffer_size = 64#batch_size*2
total_buffer_size = 0
buffer_size_max = n_sim*8000
buffer_state_pre = np.empty([buffer_size_max, (nyp-1), nzp_rl, nxp_rl, 10], dtype=np.float32)
buffer_state_stat_pre = np.empty([buffer_size_max, (nyp-1), 1, 1, 9], dtype=np.float32)
buffer_obs_pre = np.empty([buffer_size_max, (nyp-1), nzp_rl, nxp_rl, 10], dtype=np.float32)
buffer_action = np.empty([buffer_size_max, (nyp-1), nzp_rl, nxp_rl, 6], dtype=np.float32)
buffer_reward = np.empty([buffer_size_max, (nyp-1), 1], dtype=np.float32)

buffer_SL_input = np.empty([buffer_size_max, (nyp-1), nzp_rl, nxp_rl, 10], dtype=np.float32)
buffer_SL_target = np.empty([buffer_size_max, (nyp-1), nzp_rl, nxp_rl, 6], dtype=np.float32)

up_gpu, vp_gpu = get_initial()


u_ini = np.zeros([n_sim,nyp+1,nzp,nxp,2], dtype=np.float64) #horizontal u, w
v_ini = np.zeros([n_sim,nyp,nzp,nxp,1], dtype=np.float64) #vertical v
#y = np.zeros([1,nyp,1,1,1], dtype=np.float64)
#dy = np.zeros([1,nyp+1,1,1,1], dtype=np.float64)

f = h5py.File('LES_NOMODEL_Re180_32x32_initial_field.hdf5', 'r')
#y[0,:,0,0,0] = f["y"]
u_ini[:,:] = f["u"] 
v_ini[:,:] = f["v"]
f.close()

up_ini = np.fft.rfft2(u_ini, axes=(2,3))/(nzp*nxp)
vp_ini = np.fft.rfft2(v_ini, axes=(2,3))/(nzp*nxp)

up_gpu[:,:] = cp.asarray(up_ini[:]); vp_gpu[:,:] = cp.asarray(vp_ini[:])
zero_padding(up_gpu); zero_padding(vp_gpu);
buffer_up_ini_gpu[:] = up_gpu[:]; buffer_vp_ini_gpu[:] = vp_gpu[:]


critic_lr = 0.0001
actor_lr = 0.000002
Cs = 0.08
t_s[:] = 0; t_e[:] = nt;

for b in range(20):#ini_buffer_size//(2*n_sim)):

    #if t_s==0 : 
    if b%1 == 0 :
        sess.run(noise_update_op1, feed_dict={tf_ada_std : ada_std_in[0]*ada_std[0]})
        sess.run(noise_update_op2, feed_dict={tf_ada_std : ada_std_in[1]*ada_std[1]})
        sess.run(noise_update_op3, feed_dict={tf_ada_std : ada_std_in[2]*ada_std[2]})
        sess.run(noise_update_op4, feed_dict={tf_ada_std : ada_std_in[3]*ada_std[3]})
        sess.run(noise_update_op5, feed_dict={tf_ada_std : ada_std_in[4]*ada_std[4]})
        sess.run(noise_update_op6, feed_dict={tf_ada_std : ada_std_in[5]*ada_std[5]})
        sess.run(noise_update_op7, feed_dict={tf_ada_std : ada_std_in[6]*ada_std[6]})
        sess.run(noise_update_op8, feed_dict={tf_ada_std : ada_std_in[7]*ada_std[7]})
        sess.run([noise_update_actor_op1,
                  noise_update_actor_op2,
                  noise_update_actor_op3,
                  noise_update_actor_op4,
                  noise_update_actor_op5,
                  noise_update_actor_op6,
                  noise_update_actor_op7,
                  noise_update_actor_op8])                  

    for n_sim_idx in range(n_sim):
        if t_s[n_sim_idx]==0 : 
            int1 = np.random.randint(n_initial)
            up_gpu[n_sim_idx:n_sim_idx+1] = buffer_up_ini_gpu[int1:int1+1]; vp_gpu[n_sim_idx:n_sim_idx+1] = buffer_vp_ini_gpu[int1:int1+1];

    state_pre_np, obs_pre_np, action_np, \
    u_mean_pre, u_rms_pre, uv_mean_pre, dudy_mean_pre, u_mean_next, u_rms_next, uv_mean_next, dudy_mean_next, nu_t_dudy_mean_pre, nu_t_dudy_mean_next, \
    SL_input_np, SL_target_np \
    = time_loop(up_gpu,vp_gpu,SGS2p_gpu,SGSv2p_gpu,H2p_gpu,Hv2p_gpu,t_s,t_e,EQWM=False,SGSmodel=True)

    t_s[:] += nt; t_e[:] += nt

    #get only 1 point information in horizontal plane
    for to_buffer in range(1):

        buffer_idx = total_buffer_size % buffer_size_max
        int_x = 0; int_z = 0;

        SL_input_np_sub = np.reshape(SL_input_np[:,:(nyp-1),int_z:nzp_rl*nzp_r:nzp_r,int_x:nxp_rl*nxp_r:nxp_r,:], [n_sim,(nyp-1),nzp_rl,nxp_rl,10])
        SL_target_np_sub = np.reshape(SL_target_np[:,:(nyp-1),int_z:nzp_rl*nzp_r:nzp_r,int_x:nxp_rl*nxp_r:nxp_r,:], [n_sim,(nyp-1),nzp_rl,nxp_rl,6])

        buffer_SL_input[buffer_idx:buffer_idx+n_sim] = SL_input_np_sub[:]
        buffer_SL_target[buffer_idx:buffer_idx+n_sim] = SL_target_np_sub[:]
                
        total_buffer_size += n_sim


iter = -1


buffer_size = np.minimum(total_buffer_size,buffer_size_max)

input_std[:] = np.std(buffer_SL_input[:buffer_size,:,:,:,0:9], axis=(0,1,2,3), keepdims=True)
TAU_std = np.std(buffer_SL_target[:buffer_size], axis=(0,1,2,3), keepdims=True) 

fw = open('input_std', 'a')
for i in range(9):
    fw.write('%.15f\n'%(input_std[0,0,0,0,i]))
fw.close()    

for it in range(50000):

    buffer_size = np.minimum(total_buffer_size,buffer_size_max)
    int1 = np.random.randint(buffer_size, size=batch_size_SL)
    int2 = np.random.randint((nyp-1), size=batch_size_SL)
    int3 = np.random.randint(nzp_rl, size=batch_size_SL)
    int4 = np.random.randint(nxp_rl, size=batch_size_SL)
    for b in range(batch_size_SL):

        batch_SL_input[b:b+1,:,:,:,0:9] = buffer_SL_input[int1[b]:int1[b]+1,int2[b]:int2[b]+1,int3[b]:int3[b]+1,int4[b]:int4[b]+1,0:9] / input_std[:]
        batch_SL_input[b:b+1,:,:,:,9:10] = buffer_SL_input[int1[b]:int1[b]+1,int2[b]:int2[b]+1,int3[b]:int3[b]+1,int4[b]:int4[b]+1,9:10]# / input_std[:]
        batch_SL_target[b:b+1] = buffer_SL_target[int1[b]:int1[b]+1,int2[b]:int2[b]+1,int3[b]:int3[b]+1,int4[b]:int4[b]+1]
        batch_SL_weight[b:b+1] = 1.0/(TAU_std[0:1,0:1]+10**-10)

    _, SL_loss_curr = sess.run([pre_actor_opt,SL_loss], 
                         feed_dict={actor_learning_rate: 0.0001,
                                    SL_input: batch_SL_input,
                                    SL_target: batch_SL_target,
                                    SL_weight: batch_SL_weight})

    Logout = open('Pretraining_loss', 'a')
    Logout.write('%d %.10f\n'%(it, SL_loss_curr))
    Logout.close()


sess.run(noise_update_op1, feed_dict={tf_ada_std : ada_std_in[0]*ada_std[0]})
sess.run(noise_update_op2, feed_dict={tf_ada_std : ada_std_in[1]*ada_std[1]})
sess.run(noise_update_op3, feed_dict={tf_ada_std : ada_std_in[2]*ada_std[2]})
sess.run(noise_update_op4, feed_dict={tf_ada_std : ada_std_in[3]*ada_std[3]})
sess.run(noise_update_op5, feed_dict={tf_ada_std : ada_std_in[4]*ada_std[4]})
sess.run(noise_update_op6, feed_dict={tf_ada_std : ada_std_in[5]*ada_std[5]})
sess.run(noise_update_op7, feed_dict={tf_ada_std : ada_std_in[6]*ada_std[6]})
sess.run(noise_update_op8, feed_dict={tf_ada_std : ada_std_in[7]*ada_std[7]})
sess.run([noise_update_actor_op1,
          noise_update_actor_op2,
          noise_update_actor_op3,
          noise_update_actor_op4,
          noise_update_actor_op5,
          noise_update_actor_op6,
          noise_update_actor_op7,
          noise_update_actor_op8])      


N_step = 200
sub_buffer_size = 2*N_step
sub_buffer_state_pre_np = np.zeros([sub_buffer_size,n_sim,nyp-1,nzp_rl,nxp_rl,10], dtype=np.float64)
sub_buffer_obs_pre_np = np.zeros([sub_buffer_size,n_sim,nyp-1,nzp_rl,nxp_rl,10], dtype=np.float64)
sub_buffer_action_np = np.zeros([sub_buffer_size,n_sim,nyp-1,nzp_rl,nxp_rl,6], dtype=np.float64)
sub_buffer_reward_np = np.zeros([sub_buffer_size,n_sim,nyp-1], dtype=np.float64)            

sub_buffer_u_mean_pre = np.zeros([sub_buffer_size,n_sim,nyp-1,3], dtype=np.float64)
sub_buffer_u_rms_pre = np.zeros([sub_buffer_size,n_sim,nyp-1,3], dtype=np.float64)
sub_buffer_uv_mean_pre = np.zeros([sub_buffer_size,n_sim,nyp-1,1], dtype=np.float64)
sub_buffer_dudy_mean_pre = np.zeros([sub_buffer_size,n_sim,nyp-1,2], dtype=np.float64)
sub_buffer_TAU_mean_pre = np.zeros([sub_buffer_size,n_sim,nyp-1,1], dtype=np.float64)
sub_buffer_u_mean_next = np.zeros([sub_buffer_size,n_sim,nyp-1,3], dtype=np.float64)
sub_buffer_u_rms_next = np.zeros([sub_buffer_size,n_sim,nyp-1,3], dtype=np.float64)
sub_buffer_uv_mean_next = np.zeros([sub_buffer_size,n_sim,nyp-1,1], dtype=np.float64)
sub_buffer_dudy_mean_next = np.zeros([sub_buffer_size,n_sim,nyp-1,2], dtype=np.float64)
sub_buffer_TAU_mean_next = np.zeros([sub_buffer_size,n_sim,nyp-1,1], dtype=np.float64)

total_buffer_size = 0;
t_s[:] = 0; t_e[:] = nt;
global_t_s = 0;
for it in range(20000):
    
    iter += 1  
    
    # RUN SIMULATION
    #parameter space exploration

    for n_sim_idx in range(n_sim):
        if t_s[n_sim_idx]==0 : 
            int1 = np.random.randint(n_initial)
            up_gpu[n_sim_idx:n_sim_idx+1] = buffer_up_ini_gpu[int1:int1+1]; vp_gpu[n_sim_idx:n_sim_idx+1] = buffer_vp_ini_gpu[int1:int1+1];


    random_SGS_model = True
    for it_trans in range(2000):

        #actor noise update
        sess.run(noise_update_op1, feed_dict={tf_ada_std : ada_std_in[0]*ada_std[0]})
        sess.run(noise_update_op2, feed_dict={tf_ada_std : ada_std_in[1]*ada_std[1]})
        sess.run(noise_update_op3, feed_dict={tf_ada_std : ada_std_in[2]*ada_std[2]})
        sess.run(noise_update_op4, feed_dict={tf_ada_std : ada_std_in[3]*ada_std[3]})
        sess.run(noise_update_op5, feed_dict={tf_ada_std : ada_std_in[4]*ada_std[4]})
        sess.run(noise_update_op6, feed_dict={tf_ada_std : ada_std_in[5]*ada_std[5]})
        sess.run(noise_update_op7, feed_dict={tf_ada_std : ada_std_in[6]*ada_std[6]})
        sess.run(noise_update_op8, feed_dict={tf_ada_std : ada_std_in[7]*ada_std[7]})
        sess.run([noise_update_actor_op1,
                  noise_update_actor_op2,
                  noise_update_actor_op3,
                  noise_update_actor_op4,
                  noise_update_actor_op5,
                  noise_update_actor_op6,
                  noise_update_actor_op7,
                  noise_update_actor_op8])       

        sub_buffer_idx = global_t_s // nt 
        #if sub_buffer_idx < N_step : 
        state_pre_np, obs_pre_np, action_np, \
        u_mean_pre, u_rms_pre, uv_mean_pre, dudy_mean_pre, u_mean_next, u_rms_next, uv_mean_next, dudy_mean_next, nu_t_dudy_mean_pre, nu_t_dudy_mean_next, \
        SL_input_np, SL_target_np \
        = time_loop(up_gpu,vp_gpu,SGS2p_gpu,SGSv2p_gpu,H2p_gpu,Hv2p_gpu,t_s,t_e,EQWM=False,SGSmodel=False,Random_action=random_SGS_model)  

        global_t_s += nt


        int_x = 0; int_z = 0; 

        sub_buffer_state_pre_np[sub_buffer_idx%sub_buffer_size,:] = state_pre_np[:,:,int_z:nzp_rl*nzp_r:nzp_r,int_x:nxp_rl*nxp_r:nxp_r]
        sub_buffer_obs_pre_np[sub_buffer_idx%sub_buffer_size,:] = obs_pre_np[:,:,int_z:nzp_rl*nzp_r:nzp_r,int_x:nxp_rl*nxp_r:nxp_r] 
        sub_buffer_action_np[sub_buffer_idx%sub_buffer_size,:] = action_np[:,:,int_z:nzp_rl*nzp_r:nzp_r,int_x:nxp_rl*nxp_r:nxp_r] 
        #sub_buffer_reward_np[(sub_buffer_idx)%sub_buffer_size,:] = reward_np[:,:]    
        
        sub_buffer_u_mean_pre[sub_buffer_idx%sub_buffer_size,:] = u_mean_pre[:]
        sub_buffer_u_rms_pre[sub_buffer_idx%sub_buffer_size,:] = u_rms_pre[:]
        sub_buffer_uv_mean_pre[sub_buffer_idx%sub_buffer_size,:] = uv_mean_pre[:]
        sub_buffer_dudy_mean_pre[sub_buffer_idx%sub_buffer_size,:] = dudy_mean_pre[:]
        sub_buffer_TAU_mean_pre[sub_buffer_idx%sub_buffer_size,:] = nu_t_dudy_mean_pre[:]
        sub_buffer_u_mean_next[sub_buffer_idx%sub_buffer_size,:] = u_mean_next[:]
        sub_buffer_u_rms_next[sub_buffer_idx%sub_buffer_size,:] = u_rms_next[:]
        sub_buffer_uv_mean_next[sub_buffer_idx%sub_buffer_size,:] = uv_mean_next[:]
        sub_buffer_dudy_mean_next[sub_buffer_idx%sub_buffer_size,:] = dudy_mean_next[:]
        sub_buffer_TAU_mean_next[sub_buffer_idx%sub_buffer_size,:] = nu_t_dudy_mean_next[:]
        
        #--------------- uv profile
        reward_stat_uv_next = -np.abs((uv_mean_next[:,0:nyp-1,0:1] - stat_target[:,:nyp-1,7:8]))**0.5

        simulation_done = 100*reward_stat_uv_next[:,:,0]

        done_np = np.ones([n_sim,nyp-1], dtype=np.float64)
        for n_sim_idx in range(n_sim):
            for j in range(nyp-1):
                if simulation_done[n_sim_idx,j] < -100.0 or np.isnan(simulation_done[n_sim_idx,j]) or np.isinf(simulation_done[n_sim_idx,j]) : 
                    done_np[n_sim_idx,j] = 0.0;


        state_pre_np = sub_buffer_state_pre_np[(sub_buffer_idx-N_step)%sub_buffer_size,:]
        obs_pre_np = sub_buffer_obs_pre_np[(sub_buffer_idx-N_step)%sub_buffer_size,:] 
        action_np = sub_buffer_action_np[(sub_buffer_idx-N_step)%sub_buffer_size,:] 
        
        dudy_mean_next_avg = np.zeros([n_sim,nyp-1,2], dtype=np.float64)
        uv_mean_next_avg = np.zeros([n_sim,nyp-1,1], dtype=np.float64)   
        TAU_mean_next_avg = np.zeros([n_sim,nyp-1,1], dtype=np.float64)   
        u_rms_next_avg = np.zeros([n_sim,nyp-1,3], dtype=np.float64)   

        dudy_mean_next_avg[:] = sub_buffer_dudy_mean_next[(sub_buffer_idx-N_step)%sub_buffer_size,:]
        uv_mean_next_avg[:] = sub_buffer_uv_mean_next[(sub_buffer_idx-N_step)%sub_buffer_size,:]
        TAU_mean_next_avg[:] = sub_buffer_TAU_mean_next[(sub_buffer_idx-N_step)%sub_buffer_size,:]
        u_rms_next_avg[:] = sub_buffer_u_rms_next[(sub_buffer_idx-N_step)%sub_buffer_size,:]
        
        dem = 1
        for i in range(1,N_step): # 0.99 
            dudy_mean_next_avg += 0.99**i*sub_buffer_dudy_mean_next[(sub_buffer_idx-N_step+i)%sub_buffer_size,:]
            uv_mean_next_avg += 0.99**i*sub_buffer_uv_mean_next[(sub_buffer_idx-N_step+i)%sub_buffer_size,:]
            TAU_mean_next_avg += 0.99**i*sub_buffer_TAU_mean_next[(sub_buffer_idx-N_step+i)%sub_buffer_size,:]
            u_rms_next_avg += 0.99**i*sub_buffer_u_rms_next[(sub_buffer_idx-N_step+i)%sub_buffer_size,:]
            
            dem += 0.99**i
        dudy_mean_next_avg /= dem
        uv_mean_next_avg /= dem
        TAU_mean_next_avg /= dem
        u_rms_next_avg /= dem
        
        #--------------- dudy mean profile
        reward_stat_dudy_next = -np.abs((dudy_mean_next_avg[:,0:nyp-1,0:1] - stat_target[:,:nyp-1,6:7]))**0.5
        #--------------- dwdy mean profile
        reward_stat_dwdy_next = -np.abs((dudy_mean_next_avg[:,0:nyp-1,1:2] - 0.0))**0.5
        #--------------- uv profile
        reward_stat_uv_next = -np.abs((uv_mean_next_avg[:,0:nyp-1,0:1] - stat_target[:,:nyp-1,7:8]))**0.5
        #--------------- TAUxy profile
        reward_stat_TAU_next = -np.abs(TAU_mean_next_avg[:,0:nyp-1,0:1] - ((-(y[:,0:nyp-1,:,0,0] + 0.5*dy[:,1:nyp,:,0,0])) - (- stat_target[:,:nyp-1,7:8] + stat_target[:,:nyp-1,6:7])))**0.5

        reward_np = 100*reward_stat_dudy_next[:,:,0] + 0*reward_stat_dwdy_next[:,:,0] + 50*reward_stat_uv_next[:,:,0] + 0*reward_stat_TAU_next[:,:,0]
        
        u_mean_pre = sub_buffer_u_mean_pre[(sub_buffer_idx-N_step)%sub_buffer_size,:]
        u_rms_pre = sub_buffer_u_rms_pre[(sub_buffer_idx-N_step)%sub_buffer_size,:]
        uv_mean_pre = sub_buffer_uv_mean_pre[(sub_buffer_idx-N_step)%sub_buffer_size,:]
        dudy_mean_pre = sub_buffer_dudy_mean_pre[(sub_buffer_idx-N_step)%sub_buffer_size,:]

 
        for n_sim_idx in range(n_sim):
            if np.any(done_np[n_sim_idx] == 0) :
                state_pre_np[n_sim_idx] = sub_buffer_state_pre_np[(sub_buffer_idx)%sub_buffer_size,n_sim_idx]
                obs_pre_np[n_sim_idx] = sub_buffer_obs_pre_np[(sub_buffer_idx)%sub_buffer_size,n_sim_idx] 
                action_np[n_sim_idx] = sub_buffer_action_np[(sub_buffer_idx)%sub_buffer_size,n_sim_idx] 

                u_mean_pre[n_sim_idx] = sub_buffer_u_mean_pre[(sub_buffer_idx)%sub_buffer_size,n_sim_idx]
                u_rms_pre[n_sim_idx] = sub_buffer_u_rms_pre[(sub_buffer_idx)%sub_buffer_size,n_sim_idx]
                uv_mean_pre[n_sim_idx] = sub_buffer_uv_mean_pre[(sub_buffer_idx)%sub_buffer_size,n_sim_idx]
                dudy_mean_pre[n_sim_idx] = sub_buffer_dudy_mean_pre[(sub_buffer_idx)%sub_buffer_size,n_sim_idx]

                reward_np[n_sim_idx] = -100.0
                t_s[n_sim_idx] = 0; t_e[n_sim_idx] = nt;
                episode[n_sim_idx] += 1

            elif t_s[n_sim_idx] // nt < N_step :
                reward_np[n_sim_idx] = 0.0
                t_s[n_sim_idx] += nt; t_e[n_sim_idx] += nt
            else :
                t_s[n_sim_idx] += nt; t_e[n_sim_idx] += nt

        if np.any(reward_np[:, 0] != 0) : break;

    #-----------------------------
    mean_reward_np = np.zeros([nyp-1], dtype=np.float64)  
    mean_reward_dudy_np = np.zeros([nyp-1], dtype=np.float64)  
    mean_reward_dwdy_np = np.zeros([nyp-1], dtype=np.float64)  
    mean_reward_uv_np = np.zeros([nyp-1], dtype=np.float64)  
    mean_reward_TAU_np = np.zeros([nyp-1], dtype=np.float64) 
    j=0
    for n_sim_idx in range(n_sim) :
        if reward_np[n_sim_idx, 0] != 0 and reward_np[n_sim_idx, 0] != -100 :
            mean_reward_np[:] += reward_np[n_sim_idx,:]
            mean_reward_dudy_np[:] += 100*reward_stat_dudy_next[n_sim_idx,:,0]
            mean_reward_dwdy_np[:] += 100*reward_stat_dwdy_next[n_sim_idx,:,0]
            mean_reward_uv_np[:] += 100*reward_stat_uv_next[n_sim_idx,:,0]
            mean_reward_TAU_np[:] += 100*reward_stat_TAU_next[n_sim_idx,:,0]
            j += 1
    if j != 0 :
        mean_reward_np /= j 
        mean_reward_dudy_np /= j 
        mean_reward_dwdy_np /= j 
        mean_reward_uv_np /= j 
        mean_reward_TAU_np /= j 

    Filename4 = 'accumulated_reward.plt'
    Logout = open(Filename4, 'a')
    Logout.write('%d %f %f %f %f %f\n'%(it,np.mean(mean_reward_np),
                                     np.mean(mean_reward_dudy_np),np.mean(mean_reward_dwdy_np),np.mean(mean_reward_uv_np),np.mean(mean_reward_TAU_np)))
    Logout.close()


    state_stat_pre_np = np.zeros((n_sim,(nyp-1),9), dtype=np.float32)
    state_stat_pre_np[:,:,0:2] = u_mean_pre[:,0:(nyp-1),0:3:2]
    state_stat_pre_np[:,:,2:4] = dudy_mean_pre[:,0:(nyp-1),0:2]
    state_stat_pre_np[:,:,4:7] = u_rms_pre[:,0:(nyp-1),0:3]
    state_stat_pre_np[:,:,7:8] = uv_mean_pre[:,0:(nyp-1),0:1]
    state_stat_pre_np[:,:,8:9] = (1-np.abs(y[:,0:(nyp-1),0:1,0,0]+0.5*dy[:,1:(nyp-1)+1,0:1,0,0]))*Re

    #get only 1 point information in horizontal plane
    for to_buffer in range(1):
        state_pre_np_sub = np.reshape(state_pre_np[:,:(nyp-1),:,:,:], [n_sim,(nyp-1),nzp_rl,nxp_rl,10]) 
        state_stat_pre_np_sub = np.reshape(state_stat_pre_np[:,:(nyp-1)], [n_sim,(nyp-1),1,1,9])
        obs_pre_np_sub = np.reshape(obs_pre_np[:,:(nyp-1),:,:,:], [n_sim,(nyp-1),nzp_rl,nxp_rl,10])
        action_np_sub = np.reshape(action_np[:,:(nyp-1),:,:,:], [n_sim,(nyp-1),nzp_rl,nxp_rl,6])
        reward_np_sub = np.reshape(reward_np[:,:(nyp-1)], [n_sim,(nyp-1),1]) 


    for n_sim_idx in range(n_sim) :
        if reward_np[n_sim_idx, 0] != 0 :

            buffer_idx = total_buffer_size % buffer_size_max
            buffer_state_pre[buffer_idx:buffer_idx+1] = state_pre_np_sub[n_sim_idx:n_sim_idx+1]
            buffer_state_stat_pre[buffer_idx:buffer_idx+1] = state_stat_pre_np_sub[n_sim_idx:n_sim_idx+1]
            buffer_obs_pre[buffer_idx:buffer_idx+1] = obs_pre_np_sub[n_sim_idx:n_sim_idx+1]
            buffer_action[buffer_idx:buffer_idx+1] = action_np_sub[n_sim_idx:n_sim_idx+1]
            buffer_reward[buffer_idx:buffer_idx+1] = reward_np_sub[n_sim_idx:n_sim_idx+1]

            total_buffer_size += 1


    if it >= 0 :
        #replay start size
        decay_factor = 0.0
        buffer_size = np.minimum(total_buffer_size,buffer_size_max)

        if it % 1000 == 0 : 
            reward_std = np.mean(buffer_reward[:buffer_size,:,0]**2, dtype=np.float64)**0.5

            state_mean = np.mean(buffer_state_pre[:buffer_size], axis=(0,1,2,3), keepdims=True, dtype=np.float64)
            state_stat_mean = np.mean(buffer_state_stat_pre[:buffer_size], axis=(0,1,2,3), keepdims=True, dtype=np.float64)
            action_mean = np.mean(buffer_action[:buffer_size], axis=(0,1,2,3), keepdims=True, dtype=np.float64)
            
            state_std = np.std(buffer_state_pre[:buffer_size], axis=(0,1,2,3), keepdims=True, dtype=np.float64)
            state_stat_std = np.std(buffer_state_stat_pre[:buffer_size], axis=(0,1,2,3), keepdims=True, dtype=np.float64)
            action_std = np.std(buffer_action[:buffer_size], axis=(0,1,2,3), keepdims=True, dtype=np.float64)
            


        int1 = np.random.randint(buffer_size, size=batch_size)
        int2 = np.random.randint((nyp-1), size=batch_size)

        for b in range(batch_size):

            batch_state_pre[b:b+1] = buffer_state_pre[int1[b]:int1[b]+1,int2[b]:int2[b]+1]
            batch_state_stat_pre[b:b+1] = np.reshape(buffer_state_stat_pre[int1[b]:int1[b]+1,int2[b]:int2[b]+1], [1,1,1,1,9])
            batch_obs_pre[b:b+1] = buffer_obs_pre[int1[b]:int1[b]+1,int2[b]:int2[b]+1]
            batch_action[b:b+1] = buffer_action[int1[b]:int1[b]+1,int2[b]:int2[b]+1]
            batch_reward[b:b+1] = np.reshape((buffer_reward[int1[b]:int1[b]+1,int2[b]:int2[b]+1] - 0) / reward_std, [1,1,1,1,1])
            
            batch_state_action_mean[b:b+1,0:10] = state_mean[0:1,0,0,0,:]
            batch_state_action_mean[b:b+1,10:16] = action_mean[0:1,0,0,0,:]
            batch_state_action_mean[b:b+1,16:16+9] = state_stat_mean[0:1,0,0,0,:]
            
            batch_state_action_std[b:b+1,0:10] = state_std[0:1,0,0,0,:]
            batch_state_action_std[b:b+1,10:16] = action_std[0:1,0,0,0,:]
            batch_state_action_std[b:b+1,16:16+9] = state_stat_std[0:1,0,0,0,:]



        _, critic_loss_curr1, TD_loss_curr1, w_loss_curr1 \
             = sess.run([critic_opt1, critic_loss1, TD_loss1, w_loss1], 
                                 feed_dict={state_pre: batch_state_pre,
                                            state_stat_pre: batch_state_stat_pre,
                                            obs_pre: batch_obs_pre,
                                            action: batch_action,
                                            reward: batch_reward,
                                            critic_learning_rate: critic_lr,
                                            state_action_mean : batch_state_action_mean, 
                                            state_action_std : batch_state_action_std})


        _, actor_loss_curr = sess.run([actor_opt, actor_loss], 
                                     feed_dict={state_pre: batch_state_pre,
                                                state_stat_pre: batch_state_stat_pre,
                                                obs_pre: batch_obs_pre,
                                                actor_learning_rate: actor_lr,
                                                state_action_mean : batch_state_action_mean, 
                                                state_action_std : batch_state_action_std})

            
        actor_curr, \
        actor_behavior_curr1,actor_behavior_curr2,actor_behavior_curr3,actor_behavior_curr4, \
        actor_behavior_curr5,actor_behavior_curr6,actor_behavior_curr7,actor_behavior_curr8 \
             = sess.run([actor,
                         actor_behavior1,actor_behavior2,actor_behavior3,actor_behavior4,
                         actor_behavior5,actor_behavior6,actor_behavior7,actor_behavior8], 
                                 feed_dict={state_pre: batch_state_pre,
                                            state_stat_pre: batch_state_stat_pre,
                                            obs_pre: batch_obs_pre})

        noise_distance = np.zeros([n_sim], dtype=np.float64)
        noise_distance[0] = 1. - np.mean((actor_curr - np.mean(actor_curr, axis=0, keepdims=True)) / np.std(actor_curr, axis=0, keepdims=True)  
        *(actor_behavior_curr1 - np.mean(actor_behavior_curr1, axis=0, keepdims=True)) / np.std(actor_behavior_curr1, axis=0, keepdims=True))
        noise_distance[1] = 1. - np.mean((actor_curr - np.mean(actor_curr, axis=0, keepdims=True)) / np.std(actor_curr, axis=0, keepdims=True)  
        *(actor_behavior_curr2 - np.mean(actor_behavior_curr2, axis=0, keepdims=True)) / np.std(actor_behavior_curr2, axis=0, keepdims=True))
        noise_distance[2] = 1. - np.mean((actor_curr - np.mean(actor_curr, axis=0, keepdims=True)) / np.std(actor_curr, axis=0, keepdims=True)  
        *(actor_behavior_curr3 - np.mean(actor_behavior_curr3, axis=0, keepdims=True)) / np.std(actor_behavior_curr3, axis=0, keepdims=True))
        noise_distance[3] = 1. - np.mean((actor_curr - np.mean(actor_curr, axis=0, keepdims=True)) / np.std(actor_curr, axis=0, keepdims=True)  
        *(actor_behavior_curr4 - np.mean(actor_behavior_curr4, axis=0, keepdims=True)) / np.std(actor_behavior_curr4, axis=0, keepdims=True))
        noise_distance[4] = 1. - np.mean((actor_curr - np.mean(actor_curr, axis=0, keepdims=True)) / np.std(actor_curr, axis=0, keepdims=True)  
        *(actor_behavior_curr5 - np.mean(actor_behavior_curr5, axis=0, keepdims=True)) / np.std(actor_behavior_curr5, axis=0, keepdims=True))
        noise_distance[5] = 1. - np.mean((actor_curr - np.mean(actor_curr, axis=0, keepdims=True)) / np.std(actor_curr, axis=0, keepdims=True)  
        *(actor_behavior_curr6 - np.mean(actor_behavior_curr6, axis=0, keepdims=True)) / np.std(actor_behavior_curr6, axis=0, keepdims=True))
        noise_distance[6] = 1. - np.mean((actor_curr - np.mean(actor_curr, axis=0, keepdims=True)) / np.std(actor_curr, axis=0, keepdims=True)  
        *(actor_behavior_curr7 - np.mean(actor_behavior_curr7, axis=0, keepdims=True)) / np.std(actor_behavior_curr7, axis=0, keepdims=True))
        noise_distance[7] = 1. - np.mean((actor_curr - np.mean(actor_curr, axis=0, keepdims=True)) / np.std(actor_curr, axis=0, keepdims=True)  
        *(actor_behavior_curr8 - np.mean(actor_behavior_curr8, axis=0, keepdims=True)) / np.std(actor_behavior_curr8, axis=0, keepdims=True))

        noise_distance_target = 0.05
        for n_sim_idx in range(n_sim):

            if noise_distance[n_sim_idx] > noise_distance_target :
                ada_std[n_sim_idx] = ada_std[n_sim_idx]/1.01
                ada_std_in[n_sim_idx] = 0.0
            else:
                ada_std[n_sim_idx] = ada_std[n_sim_idx]*1.01   
                ada_std_in[n_sim_idx] = 1.0

    #------------------------------------------------------------------------------------------

    if iter%1000 ==0 : 
        print('Training steps: ', iter)
        Savefile = '%d/MODEL.ckpt'%(iter)
        saver.save(sess, Savefile, global_step=iter)

    if episode[0] > 50 : break
    
    
print('Training steps: ', iter)
Savefile = '%d/MODEL.ckpt'%(iter)
saver.save(sess, Savefile, global_step=iter)
