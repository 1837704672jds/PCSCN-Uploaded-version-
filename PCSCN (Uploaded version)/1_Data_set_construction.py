import os
import re
import scipy.io.wavfile
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import time
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
#####################################################
#2025.3.19
#1.Construct dataset A
#2.Extract parallel sequence features
#####################################################
high_qulity_path = 'DataSet/Dataset A/ZCHSD/high_qulity(1)'
low_qulity_path  = 'DataSet/Dataset A/ZCHSD/low_qulity(0)'
signal_step = 4
rate = 1000
################################################  定义的函数  ############################################################
def extract_number(filename):
    match = re.search(r'\d+', filename)  # 查找文件名中的所有数字
    if match:
        return int(match.group())  # 返回转换后的整数
    return float('inf')  # 如果没有找到数字，返回一个足够大的数来确保这些文件排在后面
def zero_jug(signal):
    #print(signal.shape)
    if signal.size > 0:
        signal = signal/10
        if(np.all(signal == 0)):
            zero_rate = 1
        else:
            max_positive = np.abs(np.max(signal))
            min_negative = np.abs(np.min(signal))
            zero_count = ((signal < (0.1*max_positive)) & (signal > (-0.1*min_negative))).sum()
            zero_rate = zero_count/6000
    else:
        zero_rate = 1
    return zero_rate
def Signal_normalization(signal):
    signal = signal/10
    normalization_signal = np.zeros(len(signal))

    max_positive = np.abs(np.max(signal))
    min_negative = np.abs(np.min(signal))
    if max_positive==0:max_positive=1
    if min_negative==0:min_negative=1
    normalization_signal[signal > 0] = np.abs(signal[signal > 0]) / max_positive
    normalization_signal[signal < 0] = -np.abs(signal[signal < 0]) / min_negative

    return normalization_signal
def signal_peak_get(signal):
    time = np.arange(len(signal))  # 假设time是与signal1长度相同的时间轴
    # 查找峰值
    peaks, _ = find_peaks(signal, height=None, distance=15)  # distance参数可根据数据调整 #得到的是峰值的索引
    # 插值峰值以形成包络
    # 使用线性插值，但也可以考虑其他插值方法，如cubic插值
    f = interp1d(peaks, signal[peaks], kind='linear', bounds_error=False, fill_value='extrapolate')
    # kind='linear' 线性插值
    # kind='quadratic' 二次插值
    # kind='cubic' 三次插值
    peak_envelope = f(time)

    return peak_envelope, peaks
def signal_segmentation(data_list, label_list):
    signal_list = []
    envelope_list = []
    envelope_acf_list = []
    signal_label = []
    high_quality_signals_num = 0
    low_quality_signals_num = 0
    mid_quality_signals_num = 0
    for i in range(len(data_list)):
        if label_list[i] == 0:
            loop = int((len(data_list[i]) - (signal_step * rate))//(0.5*(signal_step * rate))) #len(data_list[i])//(signal_step * rate)
            for m in range(loop):
                signal = data_list[i][int(m*0.5*signal_step * rate):int(m*0.5*signal_step*rate + signal_step*rate)]#data_list[i][m*signal_step*rate:(m+1)*signal_step*rate]
                zero_rate = zero_jug(signal)
                if zero_rate<0.9:  # 判断是不是全0序列
                    normal_signal = Signal_normalization(signal)  # 归一化
                    signal_list.append(normal_signal)  # 0 train_signal_list

                    peak_envelope, peaks = signal_peak_get(normal_signal)  # 包络和峰值
                    envelope_list.append(peak_envelope)  # 1 train_envelope_list

                    lags = signal_step * rate - 1
                    signal_envelope_acf, _ = acf(peak_envelope, nlags=lags, fft=1, alpha=0.05)
                    envelope_acf_list.append(signal_envelope_acf)  # 2 train_envelope_acf_list

                    signal_label.append(0)
                    low_quality_signals_num += 1
        elif label_list[i] == 2:
            loop = int((len(data_list[i]) - (signal_step * rate))//(0.5*(signal_step * rate)))
            for m in range(loop):
                signal = data_list[i][int(m*0.5*signal_step * rate):int(m*0.5*signal_step*rate + signal_step*rate)]
                zero_rate = zero_jug(signal)
                if zero_rate<0.9:  # 判断是不是全0序列
                    normal_signal = Signal_normalization(signal)  # 归一化
                    signal_list.append(normal_signal)  # 0 train_signal_list

                    peak_envelope, peaks = signal_peak_get(normal_signal)  # 包络和峰值
                    envelope_list.append(peak_envelope)  # 1 train_envelope_list

                    lags = signal_step * rate - 1
                    signal_envelope_acf, _ = acf(peak_envelope, nlags=lags, fft=1, alpha=0.05)
                    envelope_acf_list.append(signal_envelope_acf)  # 2 train_envelope_acf_list

                    signal_label.append(2)
                    high_quality_signals_num += 1

    array_list = []
    array_list.append(signal_list)
    array_list.append(envelope_list)
    array_list.append(envelope_acf_list)
    return array_list,signal_label,high_quality_signals_num,mid_quality_signals_num,low_quality_signals_num
########################################################################################################################
# Main function
if(1):
    #Load the data of ZCHSD and THSD into all_signal_list and all_signal_label respectively
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('The construction of Dataset A begins')
    all_signal_list = []
    all_signal_label = []
    # ZCHSD high-quality heart sounds
    if(1):
        start_time = time.time()
        ZCH_signal_num = 0
        file_list = sorted(os.listdir(high_qulity_path), key=extract_number)
        for filename in file_list:
            if filename.endswith('.wav'):
                file_path = os.path.join(high_qulity_path, filename)
                wav_file, rate = librosa.load(file_path, sr=1000)
                all_signal_list.append(wav_file)
                all_signal_label.append(2)
                ZCH_signal_num += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('ZCHSD high-quality heart sound loading time：',elapsed_time)
        print('ZCHSD high-quality heart sound individual：', ZCH_signal_num)
    # ZCHSD Low-quality heart sounds
    if(1):
        start_time = time.time()
        ZCH_signal_num = 0
        file_list = sorted(os.listdir(low_qulity_path), key=extract_number)
        for filename in file_list:
            if filename.endswith('.wav'):
                file_path = os.path.join(low_qulity_path, filename)
                wav_file, rate = librosa.load(file_path, sr=1000)
                all_signal_list.append(wav_file)
                all_signal_label.append(0)
                ZCH_signal_num += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('ZCHSD low-quality heart sound loading time：',elapsed_time)
        print('ZCHSD low-quality heart sound individual：', ZCH_signal_num)
    ########################################################################################################################
    #Load THSD
    if(1):
        TANG_path = 'Dataset/Dataset A/THSD'
        signal_TANG_list = []
        start_time = time.time()
        file_list = sorted(os.listdir(TANG_path), key=extract_number)
        for file_name in file_list:
            if file_name.endswith('.wav'):
                file_path = os.path.join(TANG_path, file_name)  # 构造完整文件路径
                rate, wav_file = scipy.io.wavfile.read(file_path)
                signal_TANG_list.append(wav_file)
        print('·····THSD has been loaded successfully·····')
    #Expand ZCHSD data with THSD data
    if(1):
        count_0 = 0
        count_2 = 0
        excel_path = 'Dataset/Dataset A/THSD/quality_label.xls'
        excel = pd.read_excel(excel_path, sheet_name='Sheet1')
        for index, row in excel.iterrows():
            if (row[1]==1 or row[1]==2 or row[1]==3):
                all_signal_list.append(signal_TANG_list[index])
                all_signal_label.append(0)
                count_0 += 1
            if row[1]==5 and count_2<900:
                all_signal_list.append(signal_TANG_list[index])
                all_signal_label.append(2)
                count_2 += 1
        print('Total number of individuals in the two datasets:：', len(all_signal_list))
        print('The total number of labels in the two datasets:：', len(all_signal_label))
        print('Total number of THSD low-quality signals:：', count_0)
        print('Total number of THSD high-quality signals：', count_2)
        print('···It is accomplished by extending the ZCHSD data with THSD data···')
    ########################################################################################################################
    #Process the data
    random_list = [0,10,50,66,100] #You can change the seeds at will. There are a total of 5 seeds
    for Seed in range(len(random_list)):
        train_signals, temp_signals, train_labels, temp_labels = train_test_split(all_signal_list, all_signal_label, test_size=0.3, random_state=random_list[Seed])
        val_signals, test_signals, val_labels, test_labels = train_test_split(temp_signals, temp_labels, test_size=0.33, random_state=random_list[Seed])#7:1:2
        if(1):#Training set
            start_time = time.time()
            train_list,train_signal_label,high_quality_signals_num,_,low_quality_signals_num = signal_segmentation(train_signals,train_labels)
            #signal_segmentation function includes the functions of segmentation, normalization, and feature extraction, and returns a data list (training data) and a label list (training labels).
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Save the training data and labels locally
            train_labels_array = np.array(train_signal_label)
            if(Seed==0):     np.save('Array/PCSCN_Training_Data/1/train_labels_array.npy', train_labels_array)# The first randomly segmented label
            elif (Seed == 1):np.save('Array/PCSCN_Training_Data/2/train_labels_array.npy', train_labels_array)
            elif (Seed == 2):np.save('Array/PCSCN_Training_Data/3/train_labels_array.npy', train_labels_array)
            elif (Seed == 3):np.save('Array/PCSCN_Training_Data/4/train_labels_array.npy', train_labels_array)
            elif (Seed == 4):np.save('Array/PCSCN_Training_Data/5/train_labels_array.npy', train_labels_array)
            train_array =  np.array(train_list)
            train_array = np.transpose(train_array, (1, 0, 2))
            if(Seed==0):     np.save('Array/PCSCN_Training_Data/1/train_array.npy', train_array)# The first randomly segmented data
            elif (Seed == 1):np.save('Array/PCSCN_Training_Data/2/train_array.npy', train_array)
            elif (Seed == 2):np.save('Array/PCSCN_Training_Data/3/train_array.npy', train_array)
            elif (Seed == 3):np.save('Array/PCSCN_Training_Data/4/train_array.npy', train_array)
            elif (Seed == 4):np.save('Array/PCSCN_Training_Data/5/train_array.npy', train_array)
            hq_num = 0
            lq_num = 0
            for i in range(len(train_labels)):
                if train_labels[i]==2:hq_num+=1
                elif train_labels[i]==0:lq_num+=1
            print('... The training set has been solved...')
            print('The number of signals processed by the training set:', len(train_signals))
            print('Number of high-quality individuals in the training set:', hq_num)
            print('Number of low-quality individuals in the training set:', lq_num)
            print('High-quality segmentation number of the training set:', high_quality_signals_num)
            print('Low-quality segmentation number of the training set:', low_quality_signals_num)
            print('Training set data shape:', train_array.shape)
            print('Training set label shape:', train_labels_array.shape)
            print(f"Training set processing time: {elapsed_time} s")
        if(1):#Validation set
            start_time = time.time()
            val_list,val_signal_label,high_quality_signals_num,_,low_quality_signals_num = signal_segmentation(val_signals,val_labels)
            end_time = time.time()
            elapsed_time = end_time - start_time

            val_labels_array = np.array(val_signal_label)
            if(Seed==0):     np.save('Array/PCSCN_Training_Data/1/val_labels_array.npy', val_labels_array)
            elif (Seed == 1):np.save('Array/PCSCN_Training_Data/2/val_labels_array.npy', val_labels_array)
            elif (Seed == 2):np.save('Array/PCSCN_Training_Data/3/val_labels_array.npy', val_labels_array)
            elif (Seed == 3):np.save('Array/PCSCN_Training_Data/4/val_labels_array.npy', val_labels_array)
            elif (Seed == 4):np.save('Array/PCSCN_Training_Data/5/val_labels_array.npy', val_labels_array)
            val_array = np.array(val_list)
            val_array = np.transpose(val_array, (1, 0, 2))
            if(Seed==0):     np.save('Array/PCSCN_Training_Data/1/val_array.npy', val_array)
            elif (Seed == 1):np.save('Array/PCSCN_Training_Data/2/val_array.npy', val_array)
            elif (Seed == 2):np.save('Array/PCSCN_Training_Data/3/val_array.npy', val_array)
            elif (Seed == 3):np.save('Array/PCSCN_Training_Data/4/val_array.npy', val_array)
            elif (Seed == 4):np.save('Array/PCSCN_Training_Data/5/val_array.npy', val_array)
            hq_num = 0
            lq_num = 0
            for i in range(len(val_labels)):
                if val_labels[i]==2:hq_num+=1
                elif val_labels[i]==0:lq_num+=1
            print('... The Validation set has been solved...')
            print('The number of signals processed by the Validation set:', len(val_signals))
            print('Number of high-quality individuals in the Validation set:', hq_num)
            print('Number of low-quality individuals in the Validation set:', lq_num)
            print('High-quality segmentation number of the Validation set:', high_quality_signals_num)
            print('Low-quality segmentation number of the Validation set:', low_quality_signals_num)
            print('Validation set data shape:', val_array.shape)
            print('Validation set label shape:', val_labels_array.shape)
            print(f"Training set processing time: {elapsed_time} s")
        if(1):#Test set
            start_time = time.time()
            test_list,test_signal_label,high_quality_signals_num,_,low_quality_signals_num = signal_segmentation(test_signals,test_labels)
            end_time = time.time()
            elapsed_time = end_time - start_time

            test_labels_array = np.array(test_signal_label)
            if(Seed==0):     np.save('Array/PCSCN_Training_Data/1/test_labels_array.npy', test_labels_array)
            elif (Seed == 1):np.save('Array/PCSCN_Training_Data/2/test_labels_array.npy', test_labels_array)
            elif (Seed == 2):np.save('Array/PCSCN_Training_Data/3/test_labels_array.npy', test_labels_array)
            elif (Seed == 3):np.save('Array/PCSCN_Training_Data/4/test_labels_array.npy', test_labels_array)
            elif (Seed == 4):np.save('Array/PCSCN_Training_Data/5/test_labels_array.npy', test_labels_array)
            test_array =  np.array(test_list)
            test_array = np.transpose(test_array, (1, 0, 2))
            if(Seed==0):     np.save('Array/PCSCN_Training_Data/1/test_array.npy', test_array)
            elif (Seed == 1):np.save('Array/PCSCN_Training_Data/2/test_array.npy', test_array)
            elif (Seed == 2):np.save('Array/PCSCN_Training_Data/3/test_array.npy', test_array)
            elif (Seed == 3):np.save('Array/PCSCN_Training_Data/4/test_array.npy', test_array)
            elif (Seed == 4):np.save('Array/PCSCN_Training_Data/5/test_array.npy', test_array)

            hq_num = 0
            lq_num = 0
            for i in range(len(test_labels)):
                if test_labels[i] == 2:
                    hq_num += 1
                elif test_labels[i] == 0:
                    lq_num += 1
            print('... The Test set has been solved...')
            print('The number of signals processed by the Test set:', len(test_signals))
            print('Number of high-quality individuals in the Test set:', hq_num)
            print('Number of low-quality individuals in the Test set:', lq_num)
            print('High-quality segmentation number of the Test set:', high_quality_signals_num)
            print('Low-quality segmentation number of the Test set:', low_quality_signals_num)
            print('Test set data shape:', test_array.shape)
            print('Test set label shape:', test_labels_array.shape)
            print(f"Training set processing time: {elapsed_time} s")
            print('**********************************************************************')
            print(f'{Seed+1}/{len(random_list)}Random seed processing is completed')
            print('**********************************************************************')
########################################################################################################################