#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:29:11 2021

Train a CNN to denoise GPS data

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gnss_tools
import scipy
from scipy import signal

train=0 # # do you want to train?
plots=1 # # do you want to make some plots?
epos=100 # how many epocs?
sr=1
epsilon=1e-6
nperseg=31
noverlap=30

print("train "+str(train))
print("plots "+str(plots))
print("epos "+str(epos))
print("sr "+str(sr))

# LOAD THE DATA
print("LOADING DATA")

# n_data = np.load('/Users/amt/Documents/sydney_dybing/GNSS-CNN/noise_samples_tunguska.npy')
# n_data=n_data.reshape(n_data.shape[0]*3,n_data.shape[1]//3)
# n_data=np.concatenate((n_data,n_data),axis=1) # this is a kludge
# x_data = np.load('/Users/amt/Documents/sydney_dybing/GNSS-CNN/rupts_0-12_NOISELESS.npy')
# x_data=x_data.reshape(x_data.shape[0]*3,x_data.shape[1]//3)
# x_data=np.concatenate((np.zeros((x_data.shape[0],64)),x_data,np.zeros((x_data.shape[0],64))),axis=1) # this is a kludge
n_data = np.load('noise.npy')
x_data = np.load('data.npy')
model_save_file="quickie.tf" 

# MAKE TRAINING AND TESTING DATA
print("MAKE TRAINING AND TESTING DATA")
# TODO: Check that this works with diego
np.random.seed(0)
siginds=np.arange(x_data.shape[0])
np.random.shuffle(siginds)
train_inds=np.sort(siginds[:int(0.9*len(siginds))])
test_inds=np.sort(siginds[int(0.9*len(siginds)):])

# # PLOT THE DATA
# if plots:
#     # plot ps to check
#     plt.figure()
#     for ii in range(x_data.shape[0]):
#         plt.plot(x_data[ii,:]) #/np.max(np.abs(x_data[ii,:]))+ii)
#     plt.title('signals')
        
#     # plot noise to check
#     plt.figure()
#     for ii in range(n_data.shape[0]):
#         plt.plot(n_data[ii,:])
#     plt.title('noise')

# do the shifts and make batches
# print("SETTING UP GENERATOR")
f, t, Zxx = signal.stft(x_data[0,:], fs=sr, nperseg=nperseg, noverlap=noverlap)
# plt.figure()
# # for istft need (x.shape[axis] - nperseg) % (nperseg-noverlap) == 0)
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# print("len(f) is "+str(len(f)))
# print("len(t) is "+str(len(t)))

# _,Zxx_inv=scipy.signal.istft(Zxx,fs=sr,nperseg=31, noverlap=30)
# plt.figure()
# plt.plot(x_data[0,:],label='original')
# plt.plot(Zxx_inv,'--',label='from inv')
# plt.legend()

# DATA GENERATOR
# x real and imaginary components of input data, y signal and noise masks
print("FIRST PASS WITH DATA GENERATOR")
my_data=gnss_tools.stft_data_generator(32,x_data,n_data,train_inds,sr,nperseg,noverlap)
x,y=next(my_data)

my_test_data=gnss_tools.stft_data_generator_valid(50,x_data,n_data,test_inds,sr,nperseg,noverlap)
x,y,sigs,noise=next(my_test_data)

# # PLOT GENERATOR RESULTS
if plots:
    for ind in range(1):
        fig, axs = plt.subplots(nrows=4,ncols=2,figsize=(10,20),sharex=True)
        t=np.arange(x[0,:,:,0].shape[1])*sr
        # plot original data and noise
        axs[0,0].plot(t,sigs[ind,:], label='signal')
        axs[0,0].plot(t,sigs[ind,:]+noise[ind,:], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
        axs[0,0].legend()
        axs[0,0].set_title('Original signal')
        axs[0,1].plot(t,noise[ind,:], label='noise')
        axs[0,1].plot(t,sigs[ind,:]+noise[ind,:], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
        axs[0,1].legend()
        axs[0,1].set_title('Original noise')
        lim=np.max(np.abs(sigs[ind,:]+noise[ind,:]))
        axs[0,0].set_ylim((-lim,lim))
        axs[0,1].set_ylim((-lim,lim))
        # plot the real and imaginary parts of the stft(signal+noise)
        axs[1,0].pcolormesh(t, f, np.abs(x[ind,:,:,0]), shading='gouraud')
        axs[1,0].set_title('Re(STFT(signal+noise))')
        axs[1,1].pcolormesh(t, f, np.abs(x[ind,:,:,1]), shading='gouraud')
        axs[1,1].set_title('Im(STFT(signal+noise))')
        # plot the output signal and noise masks
        axs[2,0].pcolormesh(t, f, y[ind,:,:,0], shading='gouraud')
        axs[2,0].set_title('Signal mask (true)')
        axs[2,1].pcolormesh(t, f, y[ind,:,:,1], shading='gouraud')  
        axs[2,1].set_title('Noise mask (true)')
        # apply masks to noisy input signal and inverse transform 
        _,tru_sig_inv=scipy.signal.istft(y[ind,:,:,0]*(x[ind,:,:,0]+x[ind,:,:,1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
        _,tru_noise_inv=scipy.signal.istft(y[ind,:,:,1]*(x[ind,:,:,0]+x[ind,:,:,1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)    
        axs[3,0].plot(t,sigs[ind,:], label='true signal')
        axs[3,0].plot(t, tru_sig_inv, alpha=0.75, color=(0.6,0,0), label='reconstructed signal')
        axs[3,0].legend()
        axs[3,0].set_title('Denoised signal')
        axs[3,0].set_ylim((-lim,lim))
        axs[3,1].plot(t,noise[ind,:], label='true noise')
        axs[3,1].plot(t, tru_noise_inv, alpha=0.75, color=(0.6,0,0), label='reconstructed noise')
        axs[3,1].set_title('Designaled noise')
        axs[3,1].set_ylim((-lim,lim))
        axs[3,1].legend()
        
# BUILD THE MODEL
print("BUILD THE MODEL")
model=gnss_tools.make_small_unet()    
        
# ADD SOME CHECKPOINTS
print("ADDING CHECKPOINTS")
checkpoint_filepath = './checks/'+model_save_file+'_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, verbose=1,
    monitor='val_acc', mode='max', save_best_only=True)

# TRAIN THE MODEL
print("TRAINING!!!")
if train:
    batch_size=32
    # if resume:
    #     print('Resuming training results from '+model_save_file)
    #     model.load_weights(checkpoint_filepath)
    # else:
    print('Training model and saving results to '+model_save_file)
        
    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file+".csv", append=True)
    # unet_tools.my_3comp_data_generator(32,x_data,n_data,sig_train_inds,noise_train_inds,sr,std)
    history=model.fit_generator(gnss_tools.stft_data_generator(batch_size,x_data,n_data,train_inds,sr),
                        steps_per_epoch=(len(train_inds))//batch_size,
                        validation_data=gnss_tools.stft_data_generator(batch_size,x_data,n_data,test_inds,sr),
                        validation_steps=(len(test_inds))//batch_size,
                        epochs=epos, callbacks=[model_checkpoint_callback,csv_logger])
    model.save_weights("./"+model_save_file)
else:
    print('Loading training results from '+model_save_file)
    model.load_weights("./"+model_save_file)
    
# plot the results
if plots:
    # training stats
    training_stats = np.genfromtxt("./"+model_save_file+'.csv', delimiter=',',skip_header=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(training_stats[:,0],training_stats[:,1])
    ax1.plot(training_stats[:,0],training_stats[:,3])
    ax1.legend(('acc','val_acc'))
    ax2.plot(training_stats[:,0],training_stats[:,2])
    ax2.plot(training_stats[:,0],training_stats[:,4])
    ax2.legend(('loss','val_loss'))
    ax2.set_xlabel('Epoch')
    ax1.set_title(model_save_file)

# MAKE SOME PREDICTIONS
my_test_data=gnss_tools.stft_data_generator_valid(25,x_data,n_data,test_inds,sr,nperseg,noverlap)
x,y,sigs,noise=next(my_test_data)
test_predictions=model.predict(x)

# # # PLOT A FEW EXAMPLES
if plots:
    for ind in range(5):
        fig, axs = plt.subplots(nrows=5,ncols=2,figsize=(10,20),sharex=True)
        t=np.arange(x[0,:,:,0].shape[1])*sr
        # plot original data and noise
        axs[0,0].plot(t,sigs[ind,:])
        axs[0,0].set_title('Original signal')
        axs[0,1].plot(t,noise[ind,:])
        axs[0,1].set_title('Original noise')
        lim=np.max(np.abs(sigs[ind,:]+noise[ind,:]))
        axs[0,0].set_ylim((-lim,lim))
        axs[0,1].set_ylim((-lim,lim))
        # plot the real and imaginary parts of the stft(signal+noise)
        axs[1,0].pcolormesh(t, f, np.abs(x[ind,:,:,0]), shading='gouraud')
        axs[1,0].set_title('Re(STFT(signal+noise))')
        axs[1,1].pcolormesh(t, f, np.abs(x[ind,:,:,1]), shading='gouraud')
        axs[1,1].set_title('Im(STFT(signal+noise))')
        # plot the output signal and noise masks
        axs[2,0].pcolormesh(t, f, np.abs(y[ind,:,:,0]), shading='gouraud')
        axs[2,0].set_title('Signal mask (true)')
        axs[2,1].pcolormesh(t, f, np.abs(y[ind,:,:,1]), shading='gouraud')  
        axs[2,1].set_title('Noise mask (true)')
        # plot the predicted signal and noise masks       
        axs[3,0].pcolormesh(t, f, np.abs(test_predictions[ind,:,:,0]), shading='gouraud')
        axs[3,0].set_title('Signal mask (predicted)')
        axs[3,1].pcolormesh(t, f, np.abs(test_predictions[ind,:,:,1]), shading='gouraud')
        axs[3,1].set_title('Noise mask (predicted)')
        # apply masks to noisy input signal and inverse transform
        _,est_sig_inv=scipy.signal.istft(test_predictions[ind,:,:,0]*(x[ind,:,:,0]+x[ind,:,:,1]*1j),fs=sr,nperseg=nperseg, noverlap=noverlap)
        _,est_noise_inv=scipy.signal.istft(test_predictions[ind,:,:,1]*(x[ind,:,:,0]+x[ind,:,:,1]*1j),fs=sr,nperseg=nperseg, noverlap=noverlap)   
        _,tru_sig_inv=scipy.signal.istft(y[ind,:,:,0]*(x[ind,:,:,0]+x[ind,:,:,1]*1j),fs=sr,nperseg=nperseg, noverlap=noverlap)
        _,tru_noise_inv=scipy.signal.istft(y[ind,:,:,1]*(x[ind,:,:,0]+x[ind,:,:,1]*1j),fs=sr,nperseg=nperseg, noverlap=noverlap)   
        axs[4,0].plot(t,sigs[ind,:],label='signal')
        axs[4,0].plot(t, est_sig_inv,label='ML')
        axs[4,0].plot(t, tru_sig_inv,label='mask')
        axs[4,0].set_title('Denoised signal')
        axs[4,0].legend(loc='lower left',mode='expand',ncol=3)
        axs[4,0].set_ylim((-lim,lim))
        axs[4,1].plot(t,noise[ind,:],label='noise')
        axs[4,1].plot(t, est_noise_inv,label='ML')
        axs[4,1].plot(t, tru_noise_inv,label='mask')
        axs[4,1].set_title('Designaled noise')
        axs[4,1].legend(loc='lower left',mode='expand',ncol=3)
        axs[4,1].set_ylim((-lim,lim))