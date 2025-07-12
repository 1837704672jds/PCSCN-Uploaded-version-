import numpy as np
from NN import *
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.metrics import confusion_matrix
def Padding_array(array):
    # 确定填充的大小
    pad_height = (224 - array.shape[2]) // 2  # 上下各填充多少
    pad_width = (224 - array.shape[3]) // 2  # 左右各填充多少
    padding = ((0, 0),  # 不在batch_size和通道数上填充
               (0, 0),  # 不在通道数上填充（因为是在高度和宽度之后填充的）
               (pad_height, pad_height+1),  # 在高度上填充
               (pad_width, pad_width))  # 在宽度上填充
    padded_array = np.pad(array, padding, mode='constant', constant_values=0)
    return padded_array  # 或者返回padded_tensor，如果你需要张量
def Model_train(Model_name ,Data_choose,Model_type,epoches = 1000,batch_size = 32,learning_rate = 0.0002):
    #Model_name    Select a specific model
    #Data_choose   Select the dataset  1=Data_Cleaning   2=Denoising   3=Do_Nothing
    #Model_type    Model type    0=Image model   1=Sequence model
    if (Data_choose == 1):  # Data_Cleaning
        if(Model_type==0):  #图像模型
            start_time = time.time()
            train_array = np.load('Array/1_Data_Cleaning/clean_train_mfccs.npy')
            train_array = train_array.transpose(0, 3, 1, 2)
            train_label = np.load('Array/1_Data_Cleaning/clean_train_label.npy')
            print('train_array:', train_array.shape)
            print('train_label:', train_label.shape)

            val_array = np.load('Array/1_Data_Cleaning/clean_val_mfccs.npy')
            val_array = val_array.transpose(0, 3, 1, 2)
            val_label = np.load('Array/1_Data_Cleaning/clean_val_label.npy')
            print('val_array:', val_array.shape)
            print('val_label:', val_label.shape)

            test_array = np.load('Array/1_Data_Cleaning/test_mfccs.npy')
            test_array = test_array.transpose(0, 3, 1, 2)
            test_label = np.load('Array/1_Data_Cleaning/test_label.npy')
            print('test_array:', test_array.shape)
            print('test_label:', test_label.shape)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for loading local data: {elapsed_time} 秒")
        elif(Model_type==1):  #序列模型
            start_time = time.time()
            train_array = np.load('Array/1_Data_Cleaning/clean_train_signal.npy')
            train_label = np.load('Array/1_Data_Cleaning/clean_train_label.npy')
            print('train_array:', train_array.shape)
            print('train_label:', train_label.shape)

            val_array = np.load('Array/1_Data_Cleaning/clean_val_signal.npy')
            val_label = np.load('Array/1_Data_Cleaning/clean_val_label.npy')
            print('val_array:', val_array.shape)
            print('val_label:', val_label.shape)

            test_array = np.load('Array/1_Data_Cleaning/test_signal.npy')
            test_label = np.load('Array/1_Data_Cleaning/test_label.npy')
            print('test_array:', test_array.shape)
            print('test_label:', test_label.shape)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for loading local data: {elapsed_time} 秒")
    elif (Data_choose == 2):  # donsied
        if (Model_type == 0):  # Image model
            start_time = time.time()
            train_array = np.load('Array/2_Denoising/denoised_train_mfccs.npy')
            train_array = train_array.transpose(0, 3, 1, 2)
            train_label = np.load('Array/2_Denoising/denoised_train_label.npy')
            print('train_array:', train_array.shape)
            print('train_label:', train_label.shape)

            val_array = np.load('Array/2_Denoising/denoised_val_mfccs.npy')
            val_array = val_array.transpose(0, 3, 1, 2)
            val_label = np.load('Array/2_Denoising/denoised_val_label.npy')
            print('val_array:', val_array.shape)
            print('val_label:', val_label.shape)

            test_array = np.load('Array/2_Denoising/test_mfccs.npy')
            test_array = test_array.transpose(0, 3, 1, 2)
            test_label = np.load('Array/2_Denoising/test_label.npy')
            print('test_array:', test_array.shape)
            print('test_label:', test_label.shape)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for loading local data: {elapsed_time} 秒")
        elif (Model_type == 1):  # Sequence model
            start_time = time.time()
            train_array = np.load('Array/2_Denoising/denoised_train_signal.npy')
            train_label = np.load('Array/2_Denoising/denoised_train_label.npy')
            print('train_array:', train_array.shape)
            print('train_label:', train_label.shape)

            val_array = np.load('Array/2_Denoising/denoised_val_signal.npy')
            val_label = np.load('Array/2_Denoising/denoised_val_label.npy')
            print('val_array:', val_array.shape)
            print('val_label:', val_label.shape)

            test_array = np.load('Array/2_Denoising/test_signal.npy')
            test_label = np.load('Array/2_Denoising/test_label.npy')
            print('test_array:', test_array.shape)
            print('test_label:', test_label.shape)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for loading local data: {elapsed_time} 秒")
    elif (Data_choose == 3):  # do nothing
        if (Model_type == 0):
            start_time = time.time()
            train_array = np.load('Array/3_Do_Nothing/train_mfccs.npy')
            train_array = train_array.transpose(0, 3, 1, 2)
            train_label = np.load('Array/3_Do_Nothing/train_label.npy')
            print('train_array:', train_array.shape)
            print('train_label:', train_label.shape)

            val_array = np.load('Array/3_Do_Nothing/val_mfccs.npy')
            val_array = val_array.transpose(0, 3, 1, 2)
            val_label = np.load('Array/3_Do_Nothing/val_label.npy')
            print('val_array:', val_array.shape)
            print('val_label:', val_label.shape)

            test_array = np.load('Array/3_Do_Nothing/test_mfccs.npy')
            test_array = test_array.transpose(0, 3, 1, 2)
            test_label = np.load('Array/3_Do_Nothing/test_label.npy')
            print('test_array:', test_array.shape)
            print('test_label:', test_label.shape)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for loading local data: {elapsed_time} 秒")
        elif (Model_type == 1):  # 序列模型
            start_time = time.time()
            train_array = np.load('Array/3_Do_Nothing/train_signal.npy')
            train_label = np.load('Array/3_Do_Nothing/train_label.npy')
            print('train_array:', train_array.shape)
            print('train_label:', train_label.shape)

            val_array = np.load('Array/3_Do_Nothing/val_signal.npy')
            val_label = np.load('Array/3_Do_Nothing/val_label.npy')
            print('val_array:', val_array.shape)
            print('val_label:', val_label.shape)

            test_array = np.load('Array/3_Do_Nothing/test_signal.npy')
            test_label = np.load('Array/3_Do_Nothing/test_label.npy')
            print('test_array:', test_array.shape)
            print('test_label:', test_label.shape)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for loading local data: {elapsed_time} 秒")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (Model_name == 'CNN_2D'):
        My_Model = CNN_2D().to(device)
    elif (Model_name == 'ResNet'):
        My_Model = ResNet().to(device)
    elif (Model_name == 'DenseNet'):
        My_Model = DenseNet().to(device)

    elif(Model_name=='CNN_1D'):
        My_Model = CNN_1D(class_num=2).to(device)
    elif(Model_name=='CNN_LSTM'):
        My_Model = CNN_LSTM().to(device)


    if(Model_type == 0):
        train_array = torch.from_numpy(train_array)
        train_label = torch.from_numpy(train_label)
        val_array = torch.from_numpy(val_array)
        val_label = torch.from_numpy(val_label)
        test_array = torch.from_numpy(test_array)
        test_label = torch.from_numpy(test_label)
    elif(Model_type == 1):
        train_array = torch.tensor(train_array,dtype=torch.float32).unsqueeze(1)
        train_label = torch.tensor(train_label,dtype=torch.float32)
        val_array = torch.tensor(val_array,dtype=torch.float32).unsqueeze(1)
        val_label = torch.tensor(val_label,dtype=torch.float32)
        test_array = torch.tensor(test_array,dtype=torch.float32).unsqueeze(1)
        test_label = torch.tensor(test_label,dtype=torch.float32)
        print('train_tensor:',train_array.shape)
        print('val_tensor:', val_array.shape)
        print('test_tensor:', test_array.shape)

    train_dataset = TensorDataset(train_array, train_label)
    val_dataset = TensorDataset(val_array, val_label)
    test_dataset = TensorDataset(test_array, test_label)


    epoches = epoches
    batch_size = batch_size
    learning_rate = learning_rate

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    My_Model = nn.DataParallel(My_Model)
    print('......My model:', Model_name, '......')
    print(My_Model)

    optimizer = torch.optim.Adam(My_Model.parameters(), lr=learning_rate, weight_decay=1e-5)  #
    loss_function = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0(低质量).9)

    best_val_loss = float('inf')
    patience = 10
    
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    
    for epoch in range(epoches):
        My_Model.train()
        train_loss = 0.0
        total = 0
        correct = 0
        # Training set training
        all_labels = []
        all_predictions = []
        for series, labels in train_loader:
            series, labels = series.to(device), labels.to(device)
            labels = labels.long()
    
            optimizer.zero_grad()
            outputs = My_Model(series)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)
        # Calculate the confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        TN, FP, FN, TP = cm.ravel()  # assuming binary classification
        # Calculate the sensitivity and specificity
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        print(f'Epoch {epoch + 1}/{epoches}')
        print(
            f'Train Loss:{train_loss:.4f}   Train Accuracy:{train_acc:.2f}%   Train Sensitivity:{sensitivity:.4f}   Train Specificity:{specificity:.4f}   Train Precision:{precision:.4f}   Train F1-Score:{f1:.4f}')
    
        # Verification set verification
        My_Model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
    
        with torch.no_grad():
            for series, labels in val_loader:
                series, labels = series.to(device), labels.to(device)
                labels = labels.long()
    
                outputs = My_Model(series)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
    
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_acc_list.append(val_acc)

        cm = confusion_matrix(all_labels, all_predictions)
        TN, FP, FN, TP = cm.ravel()  # assuming binary classification

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        print(
            f'Valid Loss:{val_loss:.4f}   Valid Accuracy:{val_acc:.2f}%   Valid Sensitivity:{sensitivity:.4f}   Valid Specificity:{specificity:.4f}   Valid Precision:{precision:.4f}   Valid F1-Score:{f1:.4f}')

        My_Model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
    
        with torch.no_grad():
            for series, labels in test_loader:
                series, labels = series.to(device), labels.to(device)
                labels = labels.long()
    
                outputs = My_Model(series)
                loss = loss_function(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
    
        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        test_acc_list.append(test_acc)

        cm = confusion_matrix(all_labels, all_predictions)
        TN, FP, FN, TP = cm.ravel()  # assuming binary classification

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        print(
            f'Test Loss:{test_loss:.4f}    Test Accuracy:{test_acc:.2f}%    Test Sensitivity:{sensitivity:.4f}    Test Specificity:{specificity:.4f}     Test Precision:{precision:.4f}   Test F1-Score:{f1:.4f}')
        if (1):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = My_Model.state_dict()
            else:
                epochs_no_improve += 1
    
            if epochs_no_improve >= patience:
                print(f'提前终止在 {epoch + 1} epoch')
                My_Model.load_state_dict(best_model_state)
                #torch.save(best_model_state, 'Model/model.pth')
                break
    
    print_model_layers_parameters(My_Model)
########################################################################################################################
#main
    #Model_name    Select a specific model
    #Data_choose   Select the dataset  1=Data_Cleaning   2=Denoising   3=Do_Nothing
    #Model_type    Model type    0=Image model   1=Sequence model
if 1: Model_train('CNN_2D', Data_choose=2, Model_type=0,batch_size=64,learning_rate=0.0005)
if 0: Model_train('ResNet', Data_choose=2, Model_type=0,batch_size=64,learning_rate=0.0005)
if 0: Model_train('DenseNet', Data_choose=2, Model_type=0,batch_size=64,learning_rate=0.0005)

if 0: Model_train('CNN_1D', Data_choose=2, Model_type=1, batch_size=64,learning_rate=0.0005)
if 0: Model_train('CNN_LSTM', Data_choose=2, Model_type=1, batch_size=64,learning_rate=0.0005)

