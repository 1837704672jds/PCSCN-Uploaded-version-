import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import time
import librosa
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import pywt
from statsmodels.tsa.stattools import acf
from NN import *

##################################################
#2025.3.19
#Clean dataset B
##################################################

########################################################################################################################
screening_rate = 1000  #Downsampling sampling rate
signal_lenth = 4*screening_rate #Signal length 4s
window_step = int(signal_lenth/2) #Window move 2s
########################################################################################################################
def extract_number(filename):
    match = re.search(r'\d+', filename)  # 查找文件名中的所有数字
    if match:
        return int(match.group())  # 返回转换后的整数
    return float('inf')  # 如果没有找到数字，返回一个足够大的数来确保这些文件排在后面
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
    time = np.arange(len(signal))
    peaks, _ = find_peaks(signal, height=None, distance=15)
    f = interp1d(peaks, signal[peaks], kind='linear', bounds_error=False, fill_value='extrapolate')
    peak_envelope = f(time)

    return peak_envelope, peaks
def get_screen_feature(signal):
    screen_feature_list = []

    envelope, peaks = signal_peak_get(signal)  # 包络和峰值

    lags = signal_lenth - 1
    envelope_acf, _ = acf(envelope, nlags=lags, fft=1, alpha=0.05)

    screen_feature_list.append(signal)
    screen_feature_list.append(envelope)
    screen_feature_list.append(envelope_acf)
    screen_feature = np.array(screen_feature_list)

    return screen_feature
def mfccs(signal):
    NFFT = 512# FFT点数
    frame_length = 128# 帧长512
    frame_step = 32 # 帧移256
    mel_filt = 80 # mel滤波器个数
    linear_filt = 80 # 线性滤波器个数
    pre_emphasis = 0.9735# 预加重系数

    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    num_frames = (len(emphasized_signal) - frame_length)//frame_step
    pad_signal_length = num_frames * frame_step + frame_length
    emphasized_signal = emphasized_signal[0:pad_signal_length]

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    # 使用之前构建的索引将信号索引为矩阵形式，
    signal_frames = emphasized_signal[indices.astype(np.int32, copy=False)]
    ham = np.hamming(frame_length)
    # 使每一帧都通过汉明窗进行处理
    frames = signal_frames * ham

    # 短时傅里叶变换
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # 计算功率谱，得到数据shape(num_frame, 257)
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (screening_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, mel_filt + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / screening_rate)# 计算滤波器组中各个滤波器的中心点
    fbank = np.zeros((mel_filt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, mel_filt + 1):
        f_m_minus = int(bin[m - 1])  # 三角滤波器左
        f_m = int(bin[m])  # 三角滤波器中
        f_m_plus = int(bin[m + 1])  # 三角滤波器右

        for k in range(f_m_minus, f_m):  # 固定行时对列进行填充
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    # 能量计算
    filter_banks1 = np.dot(pow_frames, fbank.T)
    # 防止程序报错
    filter_banks2 = np.where(filter_banks1 == 0, np.finfo(float).eps, filter_banks1)
    mfsc_feature = np.log10(filter_banks2)
    #mfsc_feature = mfsc_feature / np.max(np.abs(mfsc_feature))#归一化

    # 一阶差分计算
    mfsc_feature_deltas = np.zeros_like(mfsc_feature)
    # 中间帧
    for t in range(1, num_frames - 1, 1):
        mfsc_feature_deltas[t] = (mfsc_feature[t + 1] - mfsc_feature[t - 1]) / 2.0
    # 第一帧
    mfsc_feature_deltas[0] = (mfsc_feature[1] - mfsc_feature[0]) / 2.0
    # 最后一帧
    mfsc_feature_deltas[num_frames - 1] = (mfsc_feature[num_frames - 1] - mfsc_feature[num_frames - 2]) / 2.0
    # 归一化
    #mfsc_feature_deltas = mfsc_feature_deltas / np.max(np.abs(mfsc_feature_deltas))

    # 二阶差分计算
    mfsc_feature_deltas_deltas = np.zeros_like(mfsc_feature)
    # 中间帧
    for t in range(1, num_frames - 1, 1):
        mfsc_feature_deltas_deltas[t] = (mfsc_feature_deltas[t + 1] - mfsc_feature_deltas[t - 1]) / 2.0
    # 第一帧
    mfsc_feature_deltas_deltas[0] = (mfsc_feature_deltas[1] - mfsc_feature_deltas[0]) / 2.0
    # 最后一帧
    mfsc_feature_deltas_deltas[num_frames - 1] = (mfsc_feature_deltas[num_frames - 1] - mfsc_feature_deltas[num_frames - 2]) / 2.0

    # LFBark
    low_freq_linear = 0
    high_freq_linear = screening_rate / 2
    '''将系数段分为40个三角窗滤波器'''
    linear_hz_points = np.linspace(low_freq_linear, high_freq_linear, linear_filt + 2)
    # 计算滤波器组中各个滤波器的中心点
    bin_linear = np.floor((NFFT + 1) * linear_hz_points / screening_rate)
    # 创建一个滤波器行，短时傅里叶变换点数列的全零矩阵(40,257)
    fbank_linear = np.zeros(
        (linear_filt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, linear_filt + 1):
        f_m_minus_linear = int(bin_linear[m - 1])  # 三角滤波器左
        f_m_linear = int(bin_linear[m])  # 三角滤波器中
        f_m_plus_linear = int(bin_linear[m + 1])  # 三角滤波器右

        for k in range(f_m_minus_linear, f_m_linear):  # 固定行时对列进行填充
            fbank_linear[m - 1, k] = (k - bin_linear[m - 1]) / (bin_linear[m] - bin_linear[m - 1])
        for k in range(f_m_linear, f_m_plus_linear):
            fbank_linear[m - 1, k] = (bin_linear[m + 1] - k) / (bin_linear[m + 1] - bin_linear[m])
    # 能量计算
    filter_banks5 = np.dot(pow_frames, fbank_linear.T)
    # 防止程序报错
    filter_banks6 = np.where(filter_banks5 == 0, np.finfo(float).eps, filter_banks5)
    # 对数计算
    lfsc_feature = np.log10(filter_banks6)
    # 归一化
    #lfsc_feature = lfsc_feature / np.max(np.abs(lfsc_feature))

    # 一阶差分计算
    lfsc_feature_deltas = np.zeros_like(lfsc_feature)
    # 中间帧
    for t in range(1, num_frames - 1, 1):
        lfsc_feature_deltas[t] = (lfsc_feature[t + 1] - lfsc_feature[t - 1]) / 2.0
    # 第一帧
    lfsc_feature_deltas[0] = (lfsc_feature[1] - lfsc_feature[0]) / 2.0
    # 最后一帧
    lfsc_feature_deltas[num_frames - 1] = (lfsc_feature[num_frames - 1] - lfsc_feature[num_frames - 2]) / 2.0
    # 归一化
    #lfsc_feature_deltas = lfsc_feature_deltas / np.max(np.abs(lfsc_feature_deltas))

    # 二阶差分计算
    lfsc_feature_deltas_deltas = np.zeros_like(lfsc_feature)
    # 中间帧
    for t in range(1, num_frames - 1, 1):
        lfsc_feature_deltas_deltas[t] = (lfsc_feature_deltas[t + 1] - lfsc_feature_deltas[t - 1]) / 2.0
    # 第一帧
    lfsc_feature_deltas_deltas[0] = (lfsc_feature_deltas[1] - lfsc_feature_deltas[0]) / 2.0
    # 最后一帧
    lfsc_feature_deltas_deltas[num_frames - 1] = (lfsc_feature_deltas[num_frames - 1] - lfsc_feature_deltas[num_frames - 2]) / 2.0

    # 特征拼接
    mix_feature = np.zeros(shape=(num_frames, mel_filt, 4))
    mix_feature[:, :, 0] = mfsc_feature
    mix_feature[:, :, 1] = mfsc_feature_deltas
    mix_feature[:, :, 2] = lfsc_feature
    mix_feature[:, :, 3] = lfsc_feature_deltas
    mix_feature = mix_feature.astype(np.float32)
    return mix_feature
def wav_transform(signal):
    wavelet = 'db6'
    level = 2
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cA, cD = coeffs[0], coeffs[1:]  # 分别获取近似系数和细节系数
    # 保留原始的cD作为temp
    temp = [d.copy() for d in cD]  # 使用copy确保temp是cD的深拷贝
    # 构造只包含近似分量的信号
    cA_coeffs = [cA] + [np.zeros_like(d) for d in cD]  # 使用numpy的zeros_like创建与cD相同形状的零数组
    cA_signal = pywt.waverec(cA_coeffs, wavelet)
    # 构造只包含细节分量的信号
    #cD_coeffs = [np.zeros_like(cA)] + temp  # 将近似分量置为零
    #cD_signal = pywt.waverec(cD_coeffs, wavelet)
    return cA_signal

def get_subsignal(signal_list,label_list):
    subsignal_list = []
    subsignal_label_list = []
    normal_num = 0
    abnormal_num = 0
    for a in range(len(signal_list)):
        loop = (len(signal_list[a]) - signal_lenth) // window_step
        for b in range(loop):
            signal = signal_list[a][b * window_step:b * window_step + signal_lenth]
            signal = Signal_normalization(signal)
            subsignal_list.append(signal)
            if (label_list[a] == 1):
                subsignal_label_list.append(1)
                normal_num += 1
            elif (label_list[a] == 0):
                subsignal_label_list.append(0)
                abnormal_num += 1
    print('    subsignal_num：', len(subsignal_list))
    print('    normal_num:',normal_num)
    print('    abnormal_num:', abnormal_num)
    return subsignal_list,subsignal_label_list
def Data_Cleaning(signal_list,label_list,model_path):
    # Input signal list (unsplit) and label list
    # Output the cleaned sub-signals and tags
    # Segment and normalize the signal.
    # Extract the parallel sequence features of the signal
    # Load the quality assessment model to provide predictive labels
    # Retain the sub-signals for cleaning
    ####################################################################################################################
    subsignal_list = []
    subsignal_label_list = []
    screen_feature_list = []
    normal_num = 0
    abnormal_num = 0
    start_time1 = time.time()
    for a in range(len(signal_list)):
        loop = (len(signal_list[a])-signal_lenth) // window_step
        for b in range(loop):
            signal = signal_list[a][b*window_step:b*window_step + signal_lenth]
            signal = Signal_normalization(signal)
            subsignal_list.append(signal)
            if(label_list[a] == 1):
                subsignal_label_list.append(1)
                normal_num+=1
            elif(label_list[a] == 0):
                subsignal_label_list.append(0)
                abnormal_num += 1
            screen_array = get_screen_feature(signal) #Solve for the features used for cleaning (3,4000)
            screen_feature_list.append(screen_array)
    end_time1 = time.time() - start_time1
    print('    Preprocessing and feature extraction time：', end_time1)
    screen_feature_array = np.array(screen_feature_list)#(batch, 3, 4000)
    print('    The sub-signal segmentation is completed, and the feature solution for screening is finished')
    print('    subsignal_num：', len(subsignal_list))
    print('    normal_num:',normal_num)
    print('    abnormal_num:', abnormal_num)
    print('    Feature shapes used for screening',screen_feature_array.shape)
    ####################################################################################################################
    # 开始加载模型
    start_time1 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCSCN(class_num=2).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    end_time1 = time.time() - start_time1
    print('    Model loading time：', end_time1)
    screen_feature_tensor = torch.tensor(screen_feature_array, dtype=torch.float32)
    start_time1 = time.time()
    batch_size = 64
    num_batches = len(screen_feature_tensor) // batch_size
    if len(screen_feature_tensor) % batch_size != 0:
        num_batches += 1
    all_outputs = torch.empty(0, dtype=torch.float32, device=device)# Initialize an empty tensor to store the prediction results of all batches
    with torch.no_grad():# Disable gradient calculation
        for i in range(num_batches):
            # 计算当前批次的索引
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(screen_feature_tensor))
            batch_data = screen_feature_tensor[start_idx:end_idx].to(device)
            batch_outputs = model(batch_data)# Make predictions through models
            all_outputs = torch.cat((all_outputs, batch_outputs), dim=0)
    quality_labels = torch.argmax(all_outputs, dim=1)  # Obtain the predicted quality label
    print('    ···········')
    print('    Quality prediction completed')
    end_time1 = time.time() - start_time1
    print('    Model prediction time：', end_time1)
    print('    Predict tag shape：', quality_labels.shape)
    ####################################################################################################################
    #保留质量标签为1的信号
    clean_signal_list = []
    clean_label_list = []
    clean_normal_num = 0
    clean_abnormal_num = 0
    start_time1 = time.time()
    for a in range(len(subsignal_list)):
        if (quality_labels[a] == 1):  #Only retain high-quality sub-signals
            signal = subsignal_list[a]
            clean_signal_list.append(signal)
            if (subsignal_label_list[a] == 1):
                clean_label_list.append(1)
                clean_normal_num += 1
            elif (subsignal_label_list[a] == 0):
                clean_label_list.append(0)
                clean_abnormal_num += 1
    print('    ···········')
    print('    Data cleaning completed')
    end_time1 = time.time() - start_time1
    print('    Data deletion time：', end_time1)
    print('    clean_subsignal_num:', len(clean_signal_list))
    print('    clean_normal_num:',clean_normal_num)
    print('    clean_abnormal_num:', clean_abnormal_num)

    return clean_signal_list,clean_label_list
def Data_Denoising(signal_list,label_list):
    denoised_signal_list = []
    subsignal_list,subsignal_label_list = get_subsignal(signal_list,label_list)
    for i in range(len(subsignal_list)):
        signal = subsignal_list[i]
        denoised_signal = wav_transform(signal)
        denoised_signal_list.append(denoised_signal)
    return denoised_signal_list,subsignal_label_list
def get_mfccs(signal_list):
    mfccs_list = []
    for i in range(len(signal_list)):
        signal = signal_list[i]
        mfcc = mfccs(signal)
        mfccs_list.append(mfcc)
    return mfccs_list
########################################################################################################################
if(1):
    #加载本地数据
    normal_path = 'Dataset/Dataset B/normal'
    abnormal_path = 'Dataset/Dataset B/abnormal'
    signal_list = []
    labels_list = []
    start_time = time.time()
    file_list = sorted(os.listdir(normal_path), key=extract_number)
    for file_name in file_list:
        if file_name.endswith('.wav'):
            file_path = os.path.join(normal_path, file_name)
            wav_file, screening_rate = librosa.load(file_path, sr=screening_rate)
            signal_list.append(wav_file)
            labels_list.append(1)
    file_list = sorted(os.listdir(abnormal_path), key=extract_number)
    for file_name in file_list:
        if file_name.endswith('.wav'):
            file_path = os.path.join(abnormal_path, file_name)
            wav_file, screening_rate = librosa.load(file_path, sr=screening_rate)
            signal_list.append(wav_file)
            labels_list.append(0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('···The signal loading of the original dataset has been completed···')
    print('Number of individuals：', len(signal_list))
    print('Number of labels：', len(labels_list))
    print('Loading time：', elapsed_time)
    # Divide the dataset
    random = 0
    train_signals, temp_signals, train_labels, temp_labels = train_test_split(signal_list, labels_list, test_size=0.3,
                                                                              random_state=random)
    val_signals, test_signals, val_labels, test_labels = train_test_split(temp_signals, temp_labels, test_size=0.66,
                                                                     random_state=random)
    print('The number of individuals in the training set：',len(train_signals))
    print('The number of individuals in the validation set：', len(val_signals))
    print('Number of individuals in the test set：', len(test_signals))
    print('·································')
############################################################
# 1.Data cleaning
if(1):
    print('1 Data cleaning begins')

    print('  ················')
    print('  Perform data cleaning on the training set')
    print('  ················')
    start_time = time.time()
    clean_train_signal_list,clean_train_label_list = Data_Cleaning(train_signals,train_labels,'Model/PCSCN.pth')
    end_time = time.time() - start_time
    print('    Training set data cleaning time：', end_time)
    clean_train_signal_array = np.array(clean_train_signal_list)
    clean_train_label_array = np.array(clean_train_label_list)
    np.save('Array/1_Data_Cleaning/clean_train_signal.npy', clean_train_signal_array)
    np.save('Array/1_Data_Cleaning/clean_train_label.npy', clean_train_label_array)

    print('  ················')
    print('  Solve for mfccs on the training set')
    print('  ················')
    clean_train_mfsc_list = get_mfccs(clean_train_signal_list)
    clean_train_mfsc_array = np.array(clean_train_mfsc_list)
    print('    mfccs characteristic shape：',clean_train_mfsc_array.shape)
    np.save('Array/1_Data_Cleaning/clean_train_mfccs.npy', clean_train_mfsc_array)

    print('  ················')
    print('  Perform data cleaning on the validation set')
    print('  ················')
    start_time = time.time()
    clean_val_signal_list,clean_val_label_list = Data_Cleaning(val_signals,val_labels,'Model/PCSCN.pth')
    end_time = time.time() - start_time
    print('    Validation set data cleaning time：',end_time)
    clean_val_signal_array = np.array(clean_val_signal_list)
    clean_val_label_array = np.array(clean_val_label_list)
    np.save('Array/1_Data_Cleaning/clean_val_signal.npy', clean_val_signal_array)
    np.save('Array/1_Data_Cleaning/clean_val_label.npy', clean_val_label_array)

    print('  ················')
    print('  Solve for mfccs on the validation set')
    print('  ················')
    clean_val_mfsc_list = get_mfccs(clean_val_signal_list)
    clean_val_mfsc_array = np.array(clean_val_mfsc_list)
    print('    mfccs characteristic shape：',clean_train_mfsc_array.shape)
    np.save('Array/1_Data_Cleaning/clean_val_mfccs.npy', clean_val_mfsc_array)

    print('  ················')
    print('  Split the test set')
    print('  ················')
    test_signal_list,test_label_list = get_subsignal(test_signals,test_labels)
    test_signal_array = np.array(test_signal_list)
    test_label_array = np.array(test_label_list)
    np.save('Array/1_Data_Cleaning/test_signal.npy', test_signal_array)
    np.save('Array/1_Data_Cleaning/test_label.npy', test_label_array)
    print('  ················')
    print('  Solve for mfccs on the test set')
    print('  ················')
    test_mfsc_list = get_mfccs(test_signal_list)
    test_mfsc_array = np.array(test_mfsc_list)
    print('    mfccs characteristic shape：',test_mfsc_array.shape)
    np.save('Array/1_Data_Cleaning/test_mfccs.npy', test_mfsc_array)
# 2.Denoising
if(1):
    print('2 Data denoising begins')


    print('  ················')
    print('  Denoise the training set')
    print('  ················')
    denoised_train_signal_list, denoised_train_label_list = Data_Denoising(train_signals, train_labels)
    denoised_train_signal_array = np.array(denoised_train_signal_list)
    denoised_train_label_array = np.array(denoised_train_label_list)
    np.save('Array/2_Denoising/denoised_train_signal.npy', denoised_train_signal_array)
    np.save('Array/2_Denoising/denoised_train_label.npy', denoised_train_label_array)
    print('    The denoising of the training set has been completed')
    print('  ················')
    print('  Solve for mfccs on the training set')
    print('  ················')
    denoised_train_mfsc_list = get_mfccs(denoised_train_signal_list)
    denoised_train_mfsc_array = np.array(denoised_train_mfsc_list)
    print('    mfccs characteristic shape：', denoised_train_mfsc_array.shape)
    np.save('Array/2_Denoising/denoised_train_mfccs.npy', denoised_train_mfsc_array)

    print('  ················')
    print('  Denoise the validation set')
    print('  ················')
    denoised_val_signal_list, denoised_val_label_list = Data_Denoising(val_signals, val_labels)
    denoised_val_signal_array = np.array(denoised_val_signal_list)
    denoised_val_label_array = np.array(denoised_val_label_list)
    np.save('Array/2_Denoising/denoised_val_signal.npy', denoised_val_signal_array)
    np.save('Array/2_Denoising/denoised_val_label.npy', denoised_val_label_array)
    print('    The denoising of the validation set has been completed')
    print('  ················')
    print('  Solve for mfccs on the validation set')
    print('  ················')
    denoised_val_mfsc_list = get_mfccs(denoised_val_signal_list)
    denoised_val_mfsc_array = np.array(denoised_val_mfsc_list)
    print('    mfccs characteristic shape：', denoised_val_mfsc_array.shape)
    np.save('Array/2_Denoising/denoised_val_mfccs.npy', denoised_val_mfsc_array)

    print('  ················')
    print('  Split the test set')
    print('  ················')
    test_signal_list, test_label_list = get_subsignal(test_signals, test_labels)
    test_signal_array = np.array(test_signal_list)
    test_label_array = np.array(test_label_list)
    np.save('Array/2_Denoising/test_signal.npy', test_signal_array)
    np.save('Array/2_Denoising/test_label.npy', test_label_array)
    print('  ················')
    print('  Solve for mfccs on the test set')
    print('  ················')
    test_mfsc_list = get_mfccs(test_signal_list)
    test_mfsc_array = np.array(test_mfsc_list)
    print('    mfccs characteristic shape：', test_mfsc_array.shape)
    np.save('Array/2_Denoising/test_mfccs.npy', test_mfsc_array)
# 3.Do nothing
if(1):
    print('3 Do nothing')
    # 对训练集进行分割并求解mfccs
    print('  ················')
    print('  Segment the training set')
    print('  ················')
    train_signal_list, train_label_list = get_subsignal(train_signals, train_labels)
    train_signal_array = np.array(train_signal_list)
    train_label_array = np.array(train_label_list)
    np.save('Array/3_Do_Nothing/train_signal.npy', train_signal_array)
    np.save('Array/3_Do_Nothing/train_label.npy', train_label_array)
    print('  ················')
    print('  Solve for mfccs on the training set')
    print('  ················')
    train_mfsc_list = get_mfccs(train_signal_list)
    train_mfsc_array = np.array(train_mfsc_list)
    print('    mfccs characteristic shape：', train_mfsc_array.shape)
    np.save('Array/3_Do_Nothing/train_mfccs.npy', train_mfsc_array)
    
    # 对验证集进行分割并求解mfccs
    print('  ················')
    print('  Split the validation set')
    print('  ················')
    val_signal_list, val_label_list = get_subsignal(val_signals, val_labels)
    val_signal_array = np.array(val_signal_list)
    val_label_array = np.array(val_label_list)
    np.save('Array/3_Do_Nothing/val_signal.npy', val_signal_array)
    np.save('Array/3_Do_Nothing/val_label.npy', val_label_array)
    print('  ················')
    print('  Solve for mfccs on the validation set')
    print('  ················')
    val_mfsc_list = get_mfccs(val_signal_list)
    val_mfsc_array = np.array(val_mfsc_list)
    print('    mfccs characteristic shape：', val_mfsc_array.shape)
    np.save('Array/3_Do_Nothing/val_mfccs.npy', val_mfsc_array)
    
    # 对测试集进行分割并求解mfccs
    print('  ················')
    print('  Split the test set')
    print('  ················')
    test_signal_list, test_label_list = get_subsignal(test_signals, test_labels)
    test_signal_array = np.array(test_signal_list)
    test_label_array = np.array(test_label_list)
    np.save('Array/3_Do_Nothing/test_signal.npy', test_signal_array)
    np.save('Array/3_Do_Nothing/test_label.npy', test_label_array)
    print('  ················')
    print('  Solve for mfccs on the test set')
    print('  ················')
    test_mfsc_list = get_mfccs(test_signal_list)
    test_mfsc_array = np.array(test_mfsc_list)
    print('    mfccs characteristic shape：', test_mfsc_array.shape)
    np.save('Array/3_Do_Nothing/test_mfccs.npy', test_mfsc_array)