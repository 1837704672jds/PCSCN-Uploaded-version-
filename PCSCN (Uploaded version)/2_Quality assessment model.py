from NN import *
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.metrics import confusion_matrix

#########################################
#2025.3.19
#Load local data and train the PCSCN
#########################################
all_train_acc_list = []
all_val_acc_list = []
all_train_loss_list = []
all_val_loss_list = []
for turn in range(5):#Conduct five model training sessions
    print('**********************************************************************')
    print(f'{turn + 1}/5 Model training begins')
    print('**********************************************************************')
    #Data loading
    if 1:
        print('···Data loading begins···')
        #训练集加载
        start_time = time.time()
        if(turn == 0):   train_array = np.load('Array/PCSCN_Training_Data/1/train_array.npy')
        elif (turn == 1):train_array = np.load('Array/PCSCN_Training_Data/2/train_array.npy')
        elif (turn == 2):train_array = np.load('Array/PCSCN_Training_Data/3/train_array.npy')
        elif (turn == 3):train_array = np.load('Array/PCSCN_Training_Data/4/train_array.npy')
        elif (turn == 4):train_array = np.load('Array/PCSCN_Training_Data/5/train_array.npy')
        train_tensor =torch.tensor(train_array, dtype=torch.float32)
        print('train_tensor: ', train_tensor.shape)
        if(turn == 0):   train_labels_array = np.load('Array/PCSCN_Training_Data/1/train_labels_array.npy')
        elif (turn == 1):train_labels_array = np.load('Array/PCSCN_Training_Data/2/train_labels_array.npy')
        elif (turn == 2):train_labels_array = np.load('Array/PCSCN_Training_Data/3/train_labels_array.npy')
        elif (turn == 3):train_labels_array = np.load('Array/PCSCN_Training_Data/4/train_labels_array.npy')
        elif (turn == 4):train_labels_array = np.load('Array/PCSCN_Training_Data/5/train_labels_array.npy')
        train_labels_tensor = torch.tensor(train_labels_array, dtype=torch.float32)
        print('train_labels_tensor：', train_labels_tensor.shape)
        print('......The training set has been loaded successfully......')
        #验证集加载
        if(turn == 0):val_array = np.load('Array/PCSCN_Training_Data/1/val_array.npy')
        elif (turn == 1):val_array = np.load('Array/PCSCN_Training_Data/2/val_array.npy')
        elif (turn == 2):val_array = np.load('Array/PCSCN_Training_Data/3/val_array.npy')
        elif (turn == 3):val_array = np.load('Array/PCSCN_Training_Data/4/val_array.npy')
        elif (turn == 4):val_array = np.load('Array/PCSCN_Training_Data/5/val_array.npy')
        val_tensor = torch.tensor(val_array, dtype=torch.float32)
        print('val_tensor: ', val_tensor.shape)
        if (turn == 0):val_labels_array = np.load('Array/PCSCN_Training_Data/1/val_labels_array.npy')
        elif (turn == 1):val_labels_array = np.load('Array/PCSCN_Training_Data/2/val_labels_array.npy')
        elif (turn == 2):val_labels_array = np.load('Array/PCSCN_Training_Data/3/val_labels_array.npy')
        elif (turn == 3):val_labels_array = np.load('Array/PCSCN_Training_Data/4/val_labels_array.npy')
        elif (turn == 4):val_labels_array = np.load('Array/PCSCN_Training_Data/5/val_labels_array.npy')
        val_labels_tensor = torch.tensor(val_labels_array, dtype=torch.float32)
        print('val_labels_tensor：', val_labels_tensor.shape)
        print('......The validation set has been loaded successfully......')
        #测试集加载
        if (turn == 0):test_array = np.load('Array/PCSCN_Training_Data/1/test_array.npy')
        elif (turn == 1):test_array = np.load('Array/PCSCN_Training_Data/2/test_array.npy')
        elif (turn == 2):test_array = np.load('Array/PCSCN_Training_Data/3/test_array.npy')
        elif (turn == 3):test_array = np.load('Array/PCSCN_Training_Data/4/test_array.npy')
        elif (turn == 4):test_array = np.load('Array/PCSCN_Training_Data/5/test_array.npy')
        test_tensor = torch.tensor(test_array, dtype=torch.float32)
        print('test_tensor: ', test_tensor.shape)
        if (turn == 0):test_labels_array = np.load('Array/PCSCN_Training_Data/1/test_labels_array.npy')
        elif (turn == 1):test_labels_array = np.load('Array/PCSCN_Training_Data/2/test_labels_array.npy')
        elif (turn == 2):test_labels_array = np.load('Array/PCSCN_Training_Data/3/test_labels_array.npy')
        elif (turn == 3):test_labels_array = np.load('Array/PCSCN_Training_Data/4/test_labels_array.npy')
        elif (turn == 4):test_labels_array = np.load('Array/PCSCN_Training_Data/5/test_labels_array.npy')
        test_labels_tensor = torch.tensor(test_labels_array, dtype=torch.float32)
        print('test_labels_tensor：', test_labels_tensor.shape)
        print('......The test set has been loaded successfully......')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time for loading local data: {elapsed_time} 秒")
    #Model training
    if 1:
        # Create a dataset
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)
        # Set hyperparameters
        epoches = 1000
        batch_size = 32 #If the reader's GPU cache is insufficient, they can try to make minor adjustments to the parameters. For reference, the author's video memory is 48GB
        learning_rate = 0.0002
        # Create a data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        My_Kan = PCSCN(class_num=2).to(device)
        My_Kan = nn.DataParallel(My_Kan) # ！！If the reader has only a single graphics card, please comment out this code
        print('......My Model......')
        print(My_Kan)

        optimizer = torch.optim.Adam(My_Kan.parameters(), lr=learning_rate,weight_decay=1e-5)#
        loss_function = nn.CrossEntropyLoss()
        # Define the early stop method
        best_val_loss = float('inf')
        patience = 10  # 容忍度阈值

        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []
        test_acc_list = []
        test_loss_list = []

        for epoch in range(epoches):
            My_Kan.train()
            train_loss = 0.0
            total = 0
            correct = 0
            # 训练集训练
            all_labels = []
            all_predictions = []
            for series, labels in train_loader:
                series, labels = series.to(device), labels.to(device)
                labels = labels.long()
                labels[labels == 2] = 1.0
                labels[labels == 0] = 0.0

                optimizer.zero_grad()
                outputs = My_Kan(series)
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
            train_loss_list.append(train_loss)
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)
            TN, FP, FN, TP = cm.ravel()  # assuming binary classification
            # 计算敏感度和特异度
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            print(f'Epoch {epoch + 1}/{epoches}')
            print(f'Train Loss:{train_loss:.4f}   Train Accuracy:{train_acc:.2f}%   Train Sensitivity:{sensitivity:.4f}   Train Specificity:{specificity:.4f}   Train Precision:{precision:.4f}   Train F1-Score:{f1:.4f}')

            # 验证集验证（可选）
            My_Kan.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for series, labels in val_loader:
                    series, labels = series.to(device), labels.to(device)
                    labels = labels.long()
                    labels[labels == 2] = 1.0
                    labels[labels == 0] = 0.0

                    outputs = My_Kan(series)
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
            val_loss_list.append(val_loss)
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)
            TN, FP, FN, TP = cm.ravel()  # assuming binary classification
            # 计算敏感度和特异度
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            print(f'Valid Loss:{val_loss:.4f}   Valid Accuracy:{val_acc:.2f}%   Valid Sensitivity:{sensitivity:.4f}   Valid Specificity:{specificity:.4f}   Valid Precision:{precision:.4f}   Valid F1-Score:{f1:.4f}')

            # 测试集验证（可选）
            My_Kan.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for series, labels in test_loader:
                    series, labels = series.to(device), labels.to(device)
                    labels = labels.long()
                    labels[labels == 2] = 1.0
                    labels[labels == 0] = 0.0

                    outputs = My_Kan(series)
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
            test_loss_list.append(test_loss)
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)
            TN, FP, FN, TP = cm.ravel()  # assuming binary classification

            # 计算敏感度和特异度
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            print(f'Test Loss:{test_loss:.4f}    Test Accuracy:{test_acc:.2f}%    Test Sensitivity:{sensitivity:.4f}    Test Specificity:{specificity:.4f}     Test Precision:{precision:.4f}   Test F1-Score:{f1:.4f}')
            if(1):
                 # Early termination and model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save the state_dict of the best model
                    best_model_state = My_Kan.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f'Terminate prematurely at {epoch + 1} epoch')
                    # Use the state_dict of the best model to save the model
                    My_Kan.load_state_dict(best_model_state)
                    # Save the model locally
                    torch.save(best_model_state, 'Model/PCSCN.pth')
                    break
        print_model_layers_parameters(My_Kan)
