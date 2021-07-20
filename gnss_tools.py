#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:08:55 2020

Unet models

@author: amt
"""

import tensorflow as tf
import numpy as np
from scipy import signal

def stft_data_generator(batch_size, x_data, n_data, inds, sr, nperseg, noverlap, nlen=128):
    while True:
        # randomly select inds for the data batch
        data_batch=np.random.choice(len(inds), batch_size)
        # randomly select inds for the noise batch
        noise_batch=np.random.choice(len(inds), batch_size)
        # get range of indicies from data
        datainds=inds[data_batch]
        # get range of indicies from noise
        noiseinds=inds[noise_batch]
        # grab batch
        sig=x_data[datainds,:]
        noise=n_data[noiseinds,:]
        # calculate shifts
        time_offset=np.random.uniform(0,nlen,size=batch_size)
        # get er done
        batch_inputs=np.zeros((batch_size,16,nlen,2))
        batch_outputs=np.zeros((batch_size,16,nlen,2))
        for ii,offset in enumerate(time_offset):
            # window data, noise, and data+noise timeseries based on shift
            subsig=sig[ii,int(offset*sr):int(offset*sr)+int(nlen*sr)]
            subnoise=noise[ii,int(offset*sr):int(offset*sr)+int(nlen*sr)]
            # plt.figure()
            # plt.plot(subsig,label='signal')
            # plt.plot(subnoise,label='noise')  
            # plt.plot(subboth,label='both')    
            # plt.legend()
            # calculate stft of data, noise, and data+noise
            _, _, stftsig = signal.stft(subsig, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise = signal.stft(subnoise, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput = stftsig+stftnoise
            # batch inputs are the real and imaginary parts of the stft of signal+noise
            batch_inputs[ii,:,:,0]=np.real(stftinput)
            batch_inputs[ii,:,:,1]=np.imag(stftinput)
            # batch outputs are 
            with np.errstate(divide='ignore'):
                rat=np.nan_to_num(np.abs(stftnoise)/np.abs(stftsig),posinf=1e20)
            batch_outputs[ii,:,:,0]=1/(1+rat) # signal mask
            batch_outputs[ii,:,:,1]=rat/(1+rat) # noise mask

        yield(batch_inputs,batch_outputs)
        
def stft_data_generator_valid(batch_size, x_data, n_data, train_inds, sr, nperseg, noverlap, nlen=128):
    while True:
        # randomly select a starting index for the data batch
        data_batch=np.arange(batch_size)
        # randomly select a starting index for the noise batch
        noise_batch=np.arange(batch_size)
        # get range of indicies from data
        datainds=train_inds[data_batch]
        # get range of indicies from noise
        noiseinds=train_inds[noise_batch]
        # grab batch
        sig=x_data[datainds,:]
        noise=n_data[noiseinds,:]
        # calculate shifts
        time_offset=np.random.uniform(0,nlen,size=batch_size)
        # get er done
        batch_inputs=np.zeros((batch_size,16,nlen,2))
        batch_outputs=np.zeros((batch_size,16,nlen,4))
        subsigs=np.zeros((len(time_offset),nlen))
        subnoises=np.zeros((len(time_offset),nlen))
        for ii,offset in enumerate(time_offset):
            # window data, noise, and data+noise timeseries based on shift
            subsig=sig[ii,int(offset*sr):int(offset*sr)+int(nlen*sr)]
            subsigs[ii,:]=subsig
            subnoise=noise[ii,int(offset*sr):int(offset*sr)+int(nlen*sr)]
            subnoises[ii,:]=subnoise
            # calculate stft of data, noise, and data+noise
            _, _, stftsig = signal.stft(subsig, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise = signal.stft(subnoise, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput=stftsig+stftnoise
            # batch inputs are the real and imaginary parts of the stft of signal+noise
            batch_inputs[ii,:,:,0]=np.real(stftinput)
            batch_inputs[ii,:,:,1]=np.imag(stftinput)
            batch_outputs[ii,:,:,0]=np.real(stftsig)
            batch_outputs[ii,:,:,1]=np.imag(stftsig)
            batch_outputs[ii,:,:,2]=np.real(stftnoise)
            batch_outputs[ii,:,:,3]=np.imag(stftnoise)
            
        yield(batch_inputs,batch_outputs,subsigs,subnoises)
        
def stft_data_generator_v2(batch_size, x_data, n_data, inds, sr, nperseg, noverlap, nlen=128):
    while True:
        # randomly select inds for the data batch
        data_batch=np.random.choice(len(inds), batch_size)
        # randomly select inds for the noise batch
        noise_batch=np.random.choice(len(inds), batch_size)
        # get range of indicies from data
        datainds=inds[data_batch]
        # get range of indicies from noise
        noiseinds=inds[noise_batch]
        # grab batch
        sig=x_data[datainds,:]
        noise=n_data[noiseinds,:]
        # calculate shifts
        time_offset=np.random.uniform(0,nlen,size=batch_size)
        # get er done
        batch_inputs=np.zeros((batch_size,16,nlen,2))
        batch_outputs=np.zeros((batch_size,16,nlen,4))
        for ii,offset in enumerate(time_offset):
            # window data, noise, and data+noise timeseries based on shift
            subsig=sig[ii,int(offset*sr):int(offset*sr)+int(nlen*sr)]
            subnoise=noise[ii,int(offset*sr):int(offset*sr)+int(nlen*sr)]
            # plt.figure()
            # plt.plot(subsig,label='signal')
            # plt.plot(subnoise,label='noise')  
            # plt.plot(subboth,label='both')    
            # plt.legend()
            # calculate stft of data, noise, and data+noise
            _, _, stftsig = signal.stft(subsig, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise = signal.stft(subnoise, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput = stftsig+stftnoise
            # batch inputs are the real and imaginary parts of the stft of signal+noise
            batch_inputs[ii,:,:,0]=np.real(stftinput)
            batch_inputs[ii,:,:,1]=np.imag(stftinput)
            # batch inputs are the real and imaginary parts of the stft of signal+noise
            batch_outputs[ii,:,:,0]=np.real(stftsig)
            batch_outputs[ii,:,:,1]=np.imag(stftsig)
            batch_outputs[ii,:,:,2]=np.real(stftnoise)
            batch_outputs[ii,:,:,3]=np.imag(stftnoise)

        yield(batch_inputs,batch_outputs)
        
def stft_data_generator_v2_valid(batch_size, x_data, n_data, train_inds, sr, nperseg, noverlap, nlen=128):
    while True:
        # randomly select a starting index for the data batch
        data_batch=np.arange(batch_size)
        # randomly select a starting index for the noise batch
        noise_batch=np.arange(batch_size)
        # get range of indicies from data
        datainds=train_inds[data_batch]
        # get range of indicies from noise
        noiseinds=train_inds[noise_batch]
        # grab batch
        sig=x_data[datainds,:]
        noise=n_data[noiseinds,:]
        # calculate shifts
        time_offset=np.random.uniform(0,nlen,size=batch_size)
        # get er done
        batch_inputs=np.zeros((batch_size,16,nlen,2))
        batch_outputs=np.zeros((batch_size,16,nlen,4))
        subsigs=np.zeros((len(time_offset),nlen))
        subnoises=np.zeros((len(time_offset),nlen))
        for ii,offset in enumerate(time_offset):
            # window data, noise, and data+noise timeseries based on shift
            subsig=sig[ii,int(offset*sr):int(offset*sr)+int(nlen*sr)]
            subnoise=noise[ii,int(offset*sr):int(offset*sr)+int(nlen*sr)]
            subsigs[ii,:]=subsig
            subnoises[ii,:]=subnoise
            # plt.figure()
            # plt.plot(subsig,label='signal')
            # plt.plot(subnoise,label='noise')  
            # plt.plot(subboth,label='both')    
            # plt.legend()
            # calculate stft of data, noise, and data+noise
            _, _, stftsig = signal.stft(subsig, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise = signal.stft(subnoise, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput = stftsig+stftnoise
            # batch inputs are the real and imaginary parts of the stft of signal+noise
            batch_inputs[ii,:,:,0]=np.real(stftinput)
            batch_inputs[ii,:,:,1]=np.imag(stftinput)
            # batch inputs are the real and imaginary parts of the stft of signal+noise
            batch_outputs[ii,:,:,0]=np.real(stftsig)
            batch_outputs[ii,:,:,1]=np.imag(stftsig)
            batch_outputs[ii,:,:,2]=np.real(stftnoise)
            batch_outputs[ii,:,:,3]=np.imag(stftnoise)

        yield(batch_inputs,batch_outputs,subsigs,subnoises)
        
def make_unet(drop=0):
    
    input_layer=tf.keras.layers.Input(shape=(32,128,2)) # 1 Channel seismic data
    
    # 1st level -- blue arrow
    level1=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(input_layer) # N filters, kernel Size, Strides, padding
    level1=tf.keras.layers.BatchNormalization()(level1)
    network=level1
    
    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(level1) # N filters, kernel Size, Strides, padding
    level1b=tf.keras.layers.BatchNormalization()(level1b)
    level1b=tf.keras.layers.Dropout(drop)(level1b)
    network=level1b
    
    # 1st --> 2nd level -- orange arrow
    level2=tf.keras.layers.Conv2D(8,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level2=tf.keras.layers.BatchNormalization()(level2)
    
    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(level2) # N filters, kernel Size, Strides, padding
    level2b=tf.keras.layers.BatchNormalization()(level2b)
    level2b=tf.keras.layers.Dropout(drop)(level2b)
    network=level2b
    
    # 2nd --> 3rd level -- orange arrow
    level3=tf.keras.layers.Conv2D(16,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level3=tf.keras.layers.BatchNormalization()(level3)
    
    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(level3) # N filters, kernel Size, Strides, padding
    level3b=tf.keras.layers.BatchNormalization()(level3b)
    level3b=tf.keras.layers.Dropout(drop)(level3b)
    network=level3b
    
    # 3rd --> 4th level -- orange arrow
    level4=tf.keras.layers.Conv2D(32,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level4=tf.keras.layers.BatchNormalization()(level4)
    
    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(level4) # N filters, kernel Size, Strides, padding
    level4b=tf.keras.layers.BatchNormalization()(level4b)  
    level4b=tf.keras.layers.Dropout(drop)(level4b)
    network=level4b
    
    # 4th --> 5th level -- orange arrow
    level5=tf.keras.layers.Conv2D(64,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level5=tf.keras.layers.BatchNormalization()(level5)
    
    # 5th level -- blue arrow
    level5b=tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(level5) # N filters, kernel Size, Strides, padding
    level5b=tf.keras.layers.BatchNormalization()(level5b)
    level5b=tf.keras.layers.Dropout(drop)(level5b)
    network=level5b

    # 5th level --> 6th level -- orange arrow
    level6=tf.keras.layers.Conv2D(128,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level6=tf.keras.layers.BatchNormalization()(level6)
    
    # 6th level -- blue arrow
    level6b=tf.keras.layers.Conv2D(256,3,activation='relu',padding='same')(level6) # N filters, kernel Size, Strides, padding
    level6b=tf.keras.layers.BatchNormalization()(level6b) 
    level6b=tf.keras.layers.Dropout(drop)(level6b)
    network=level6b
    
    # # #Base of Network

    
    # 6th level --> 5th level -- green arrow + skip connection
    level5u=tf.keras.layers.Conv2DTranspose(128,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level5u) 
    network=tf.keras.layers.Concatenate()([network,level5b])

    # 5th level -- blue arrow
    level5b=tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level5b) 
    
    # 5th level --> 4th level -- green arrow + skip connection
    level4u=tf.keras.layers.Conv2DTranspose(64,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level4u) 
    network=tf.keras.layers.Concatenate()([network,level4b])

    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level4b)     
    
    # 4th level --> 3rd level -- green arrow + skip connection
    level3u=tf.keras.layers.Conv2DTranspose(32,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level3u) 
    network=tf.keras.layers.Concatenate()([network,level3b])

    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level3b)  
    
    # 3rd level --> 2nd level -- green arrow + skip connection
    level2u=tf.keras.layers.Conv2DTranspose(16,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level2u) 
    network=tf.keras.layers.Concatenate()([network,level2b])

    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level2b) 
    
    # 2nd level --> 1st level -- green arrow + skip connection
    level1u=tf.keras.layers.Conv2DTranspose(8,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level1u) 
    network=tf.keras.layers.Concatenate()([network,level1b])

    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level1b) 
    
    # End of network
    output = tf.keras.layers.Conv2D(2,1,activation='softmax',padding='same')(network)# N filters, Filter Size, Stride, padding
    
    model=tf.keras.models.Model(input_layer,output)
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file='denoising_model_plot.png', show_shapes=True, show_layer_names=True)
    
    return model

def make_small_unet(drop=0):
    
    input_layer=tf.keras.layers.Input(shape=(16,128,2)) # 1 Channel seismic data
    
    # 1st level -- blue arrow
    level1=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(input_layer) # N filters, kernel Size, Strides, padding
    level1=tf.keras.layers.BatchNormalization()(level1)
    network=level1
    
    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(level1) # N filters, kernel Size, Strides, padding
    level1b=tf.keras.layers.BatchNormalization()(level1b)
    level1b=tf.keras.layers.Dropout(drop)(level1b)
    network=level1b
    
    # 1st --> 2nd level -- orange arrow
    level2=tf.keras.layers.Conv2D(8,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level2=tf.keras.layers.BatchNormalization()(level2)
    
    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(level2) # N filters, kernel Size, Strides, padding
    level2b=tf.keras.layers.BatchNormalization()(level2b)
    level2b=tf.keras.layers.Dropout(drop)(level2b)
    network=level2b
    
    # 2nd --> 3rd level -- orange arrow
    level3=tf.keras.layers.Conv2D(16,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level3=tf.keras.layers.BatchNormalization()(level3)
    
    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(level3) # N filters, kernel Size, Strides, padding
    level3b=tf.keras.layers.BatchNormalization()(level3b)
    level3b=tf.keras.layers.Dropout(drop)(level3b)
    network=level3b
    
    # 3rd --> 4th level -- orange arrow
    level4=tf.keras.layers.Conv2D(32,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level4=tf.keras.layers.BatchNormalization()(level4)
    
    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(level4) # N filters, kernel Size, Strides, padding
    level4b=tf.keras.layers.BatchNormalization()(level4b)  
    level4b=tf.keras.layers.Dropout(drop)(level4b)
    network=level4b
    
    # 4th --> 5th level -- orange arrow
    level5=tf.keras.layers.Conv2D(64,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level5=tf.keras.layers.BatchNormalization()(level5)
    
    # 5th level -- blue arrow
    level5b=tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(level5) # N filters, kernel Size, Strides, padding
    level5b=tf.keras.layers.BatchNormalization()(level5b)
    level5b=tf.keras.layers.Dropout(drop)(level5b)
    network=level5b
    
    # # #Base of Network
    
    # 5th level --> 4th level -- green arrow + skip connection
    level4u=tf.keras.layers.Conv2DTranspose(64,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level4u) 
    network=tf.keras.layers.Concatenate()([network,level4b])

    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level4b)     
    
    # 4th level --> 3rd level -- green arrow + skip connection
    level3u=tf.keras.layers.Conv2DTranspose(32,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level3u) 
    network=tf.keras.layers.Concatenate()([network,level3b])

    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level3b)  
    
    # 3rd level --> 2nd level -- green arrow + skip connection
    level2u=tf.keras.layers.Conv2DTranspose(16,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level2u) 
    network=tf.keras.layers.Concatenate()([network,level2b])

    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level2b) 
    
    # 2nd level --> 1st level -- green arrow + skip connection
    level1u=tf.keras.layers.Conv2DTranspose(8,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level1u) 
    network=tf.keras.layers.Concatenate()([network,level1b])

    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level1b) 
    
    # End of network
    output = tf.keras.layers.Conv2D(2,1,activation='softmax',padding='same')(network)# N filters, Filter Size, Stride, padding
    
    model=tf.keras.models.Model(input_layer,output)
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file='denoising_model_plot.png', show_shapes=True, show_layer_names=True)
    
    return model

def make_small_unet_v2(drop=0):
    
    input_layer=tf.keras.layers.Input(shape=(16,128,2)) # 1 Channel seismic data
    
    # 1st level -- blue arrow
    level1=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(input_layer) # N filters, kernel Size, Strides, padding
    level1=tf.keras.layers.BatchNormalization()(level1)
    network=level1
    
    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(level1) # N filters, kernel Size, Strides, padding
    level1b=tf.keras.layers.BatchNormalization()(level1b)
    level1b=tf.keras.layers.Dropout(drop)(level1b)
    network=level1b
    
    # 1st --> 2nd level -- orange arrow
    level2=tf.keras.layers.Conv2D(8,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level2=tf.keras.layers.BatchNormalization()(level2)
    
    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(level2) # N filters, kernel Size, Strides, padding
    level2b=tf.keras.layers.BatchNormalization()(level2b)
    level2b=tf.keras.layers.Dropout(drop)(level2b)
    network=level2b
    
    # 2nd --> 3rd level -- orange arrow
    level3=tf.keras.layers.Conv2D(16,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level3=tf.keras.layers.BatchNormalization()(level3)
    
    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(level3) # N filters, kernel Size, Strides, padding
    level3b=tf.keras.layers.BatchNormalization()(level3b)
    level3b=tf.keras.layers.Dropout(drop)(level3b)
    network=level3b
    
    # 3rd --> 4th level -- orange arrow
    level4=tf.keras.layers.Conv2D(32,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level4=tf.keras.layers.BatchNormalization()(level4)
    
    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(level4) # N filters, kernel Size, Strides, padding
    level4b=tf.keras.layers.BatchNormalization()(level4b)  
    level4b=tf.keras.layers.Dropout(drop)(level4b)
    network=level4b
    
    # 4th --> 5th level -- orange arrow
    level5=tf.keras.layers.Conv2D(64,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level5=tf.keras.layers.BatchNormalization()(level5)
    
    # 5th level -- blue arrow
    level5b=tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(level5) # N filters, kernel Size, Strides, padding
    level5b=tf.keras.layers.BatchNormalization()(level5b)
    level5b=tf.keras.layers.Dropout(drop)(level5b)
    network=level5b
    
    # # #Base of Network
    
    # 5th level --> 4th level -- green arrow + skip connection
    level4u=tf.keras.layers.Conv2DTranspose(64,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level4u) 
    network=tf.keras.layers.Concatenate()([network,level4b])

    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level4b)     
    
    # 4th level --> 3rd level -- green arrow + skip connection
    level3u=tf.keras.layers.Conv2DTranspose(32,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level3u) 
    network=tf.keras.layers.Concatenate()([network,level3b])

    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level3b)  
    
    # 3rd level --> 2nd level -- green arrow + skip connection
    level2u=tf.keras.layers.Conv2DTranspose(16,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level2u) 
    network=tf.keras.layers.Concatenate()([network,level2b])

    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level2b) 
    
    # 2nd level --> 1st level -- green arrow + skip connection
    level1u=tf.keras.layers.Conv2DTranspose(8,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level1u) 
    network=tf.keras.layers.Concatenate()([network,level1b])

    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level1b) 
    
    # End of network
    # this is the logarithm of the amplitude ratio
    output = tf.keras.layers.Conv2D(4,1,activation='linear',padding='same')(network)# N filters, Filter Size, Stride, padding    
    model=tf.keras.models.Model(input_layer,output)
    
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss=['mse'],optimizer=opt,metrics=['accuracy','mse'])
    
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file='denoising_model_plot.png', show_shapes=True, show_layer_names=True)
    
    return model

def main():
    make_small_unet_v2()

if __name__ == "__main__":
    main()