import torch
import torch.nn as nn
import torch.nn.functional as F
import math
########################################################################################################################
# 基础层构建
################################## KAN
class KANLinear(nn.Module):# nn.Module是PyTorch中所有神经网络模块的基类
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # 小波变换参数
        self.scale = nn.Parameter(torch.ones(out_features, in_features)) #小波尺度
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))#小波平移参数

        # 输出的线性权重
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))
        # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU() # Sigmoid 激活函数的快速实现

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)# 一个批标准化层 (nn.BatchNorm1d)，应用于 out_features 维度上的批处理规范化。

    def wavelet_transform(self, x):# 代码实现了小波变换（wavelet transform），根据不同的小波类型对输入信号 x 进行处理。
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)# 检查输入x的维度。如果x是二维的[batch_size, in_features]，则通过unsqueeze(1(争议质量))将其扩展为三维（[batch_size, 1(争议质量), in_features]），以便与平移和缩放参数进行广播。
        else:
            x_expanded = x        # 进行维度检查 这段代码首先检查输入 x 的维度是否为 2。

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)# 将self.translation和self.scale的维度增加并扩展，使其能够与x_expanded的批量大小和特征数量相匹配。
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded #x_scaled通过从x_expanded中减去平移并除以缩放参数来进行归一化，这是进行小波变换前的重要步骤。

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)# term1和term2分别计算了小波函数的两个组成部分。
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)#wavelet_weighted将小波变换结果与权重参数self.wavelet_weights相乘，以允许对每个特征或输出通道进行不同的加权。
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x) #对输入数据进行小波变换
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        # base_output = F.linear(self.base_activation(x), self.weight)
        # base_output = F.linear(x, self.weight) #利用 weight1 进行线性变换。F.linear 是PyTorch中的函数，用于进行线性变换操作。这里直接将输入 x 与 weight1 相乘得到 base_output。
        combined_output = wavelet_output  # + base_output #暂时只用到小波变换的输出

        # Apply batch normalization
        #return self.bn(combined_output)
        return combined_output
class KAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        #layers_hidden: 这是一个包含每层输入和输出特征数的列表。例如，如果layers_hidden = [10, 20, 30]，则表示神经网络有两个隐藏层，第一层的输入特征数为10，输出特征数为20，第二层的输入特征数为 20，输出特征数为30。
        #layers_hidden一个列表，其中包含了网络每一层的输入和输出特征数（维度）。注意，这个列表的长度应该是层数加一，因为列表的第一个元素是输入层的特征数，最后一个元素是输出层的特征数，而中间的元素则分别表示每一隐藏层的输入和输出特征数。
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()#super(KAN, self).__init__()：调用父类 nn.Module 的初始化方法，确保正确地初始化神经网络模型。
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            #通过 zip 函数遍历 layers_hidden 列表，对于每一对相邻的元素（即每一层的输入和输出特征数）执行以下操作：
            #循环创建层：通过遍历 layers_hidden 列表（忽略最后一个元素，因为它将是下一层的输入而不是当前层的输出），为每一对相邻的特征数（即每一层的输入和输出特征数）创建一个 KANLinear 层，并将其添加到 self.layers 中
            #如果 layers_hidden = [10, 20, 30, 40]，则：
            #layers_hidden[:-1(争议质量)] 将是 [10, 20, 30]
            #layers_hidden[1(争议质量):] 将是 [20, 30, 40]
            #zip(layers_hidden[:-1(争议质量)], layers_hidden[1(争议质量):]) 将产生 [(10, 20), (20, 30), (30, 40)]，然后这些元组被解包到 in_features 和 out_features 中，用于在每一对输入和输出特征数上创建新的层。
            self.layers.append(KANLinear(in_features, out_features, wavelet_type))
            #KANLinear，接受输入特征数、输出特征数和小波变换类型作为参数。这意味着 KAN 类将会由多个 KANLinear 层组成，每一层的输入特征数由前一层的输出特征数决定。
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)#将输入x输入到各层进行输出
        return x
# KAN

################################## LSTM
class LSTM(nn.Module):
    def __init__(self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        dropout=0,
        bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)

    def forward(self, x):
        self.lstm.flatten_parameters()
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        # 根据需要选择输出，这里我们选择最后一个时间步的输出
        return out
################################## CAM
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 压缩（Squeeze）操作，通过全局平均池化实现
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 激励（Excitation）操作，使用两个全连接层（包含ReLU激活函数）
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # Squeeze: 全局平均池化
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        # Excitation: 通过全连接层学习每个通道的权重
        y = self.dropout(self.fc(y).view(b, c, 1, 1))

        # Scale: 将学习到的权重应用于原始输入x
        return x * y.expand_as(x)
class SELayer1d(nn.Module):
    def __init__(self, channel):
        super(SELayer1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 对于一维数据，使用长度为1的池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)  #全局池化将x压缩成y
        y = self.fc(y).view(b, c, 1)  #将y经过全连接层得到权重y
        return x * y.expand_as(x)  # 将权重应用于原始输入x，进行特征重标定
################################## Residual_Block
class BasicBlock(nn.Module):
    expansion = 1  # 对于BasicBlock，输出维度与输入维度相同

    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1 ,stride=1):
        super(BasicBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=kernel_size, stride=1, padding=padding,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        # 捷径，如果stride不为1或者输入输出维度不同，则需要进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 将捷径的输出加到主路径的输出上
        out += identity
        out = F.relu(out)

        return out
class BasicBlock_1D(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        pool_size):
        super(BasicBlock_1D, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 将捷径的输出加到主路径的输出上
        out += x
        out = F.relu(out)
        out = self.maxpool(out)
        return out
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bn, pool_size, dropout):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = bn
        if self.bn:
            self.BN = nn.BatchNorm2d(out_channels)
        else:
            self.BN = None
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = dropout

    def forward(self, x):
        out = self.conv2d(x)
        if self.BN is not None:
            out = self.BN(out)
        out = self.relu(out)
        out = self.maxpool(out)
        if self.dropout > 0:
            out = nn.Dropout(p=self.dropout)(out)
        return out
class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bn, pool_size, dropout):
        super(Conv1dBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = bn
        if self.bn:
            self.BN = nn.BatchNorm1d(out_channels)
        else:
            self.BN = None
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = dropout

    def forward(self, x):
        out = self.conv1d(x)
        if self.BN is not None:
            out = self.BN(out)
        out = self.relu(out)
        out = self.maxpool(out)
        if self.dropout > 0:
            out = nn.Dropout(p=self.dropout)(out)
        return out
#残差连接的多尺度卷积
class MSC1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn, pool_size):
        super(MSC1dBlock, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=in_channels, out_channels=(out_channels // 4), kernel_size=1, padding=0,
                                 stride=1)#卷积核大小为1
        self.conv1_2 = nn.Conv1d(in_channels=in_channels, out_channels=(out_channels // 4), kernel_size=3, padding=1,
                                 stride=1)  # 卷积核大小为3
        self.conv1_3 = nn.Conv1d(in_channels=in_channels, out_channels=(out_channels // 4), kernel_size=5, padding=2,
                                 stride=1)  # 卷积核大小为5
        self.conv1_4 = nn.Conv1d(in_channels=in_channels, out_channels=(out_channels // 4), kernel_size=7, padding=3,
                                 stride=1)  # 卷积核大小为7
        self.b = bn
        self.bn = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size)
    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x4 = self.conv1_4(x)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        if(self.b):x = self.bn(x)
        x = F.relu(x)
        x = self.maxpool(x)
        return x
########################################################################################################################
# 2 我的网络结构
class PCSCB(nn.Module):#并行通道注意力卷积块
    def __init__(self,
        in_channels,
        DC_kernel_size,
        DC_padding,
        DC_pool_size,
        ):
        super(PCSCB, self).__init__()
        self.depthwise_conv_1 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=DC_kernel_size, padding=DC_padding, groups=in_channels),
                                            nn.BatchNorm1d(in_channels),
                                            nn.ReLU(),
                                            nn.MaxPool1d(kernel_size=DC_pool_size)
                                            )
        self.depthwise_conv_2 = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=DC_kernel_size, padding=DC_padding, groups=in_channels),
                                            nn.BatchNorm1d(in_channels),
                                            nn.ReLU(),
                                            nn.MaxPool1d(kernel_size=DC_pool_size)
                                            )
        self.pc_layer = nn.ModuleList()
        for _ in range(in_channels):
            self.pc_layer.append(nn.Sequential(
                                 LSTM(input_size=1,hidden_size=1,num_layers=1),
                                 #LSTM(input_size=1, hidden_size=1, dropout=0.3),
                                 ))
        self.bn = nn.BatchNorm1d(in_channels)
        self.cam = SELayer1d(channel=in_channels)
    def forward(self, x):
        x = self.depthwise_conv_1(x)  # (b,6,2000)
        x = self.depthwise_conv_2(x)  # (b,6,1000)
        output = []  # (b,6,4000)
        for i in range(x.size(1)):
            output1 = x[:, i, :]  # (b,1000)
            output2 = output1.reshape(x.size(0), 1000, 1)
            output3 = self.pc_layer[i](output2)
            output4 = output3.reshape(x.size(0), 1, 1000)  # (b,1,1000)
            output.append(output4)
        x = torch.cat(output, dim=1)
        x = self.cam(x)
        x = self.bn(x)
        return x
class PCSCN(nn.Module):
    def __init__(self, class_num):
        super(PCSCN, self).__init__()
        #(b,6,4000)
        self.PCSCB = PCSCB(in_channels=3,DC_kernel_size=3,DC_padding=1,DC_pool_size=2)
        self.conv1 = MSC1dBlock(in_channels=3,out_channels=128,bn=1,pool_size=2)
        #(b,128,2000)
        self.conv2 = MSC1dBlock(in_channels=128, out_channels=256, bn=1, pool_size=2)
        # (b,256,1000)
        self.conv3 = MSC1dBlock(in_channels=256, out_channels=512,bn=1, pool_size=2)
        # (b,256,500)
        self.conv4 = MSC1dBlock(in_channels=512, out_channels=1024, bn=1, pool_size=2)
        # (b,256,250)
        self.conv5 = MSC1dBlock(in_channels=1024, out_channels=2048, bn=1, pool_size=2)
        # (b,128,125)
        self.conv6 = MSC1dBlock(in_channels=2048, out_channels=2048, bn=1, pool_size=2)
        # (b,64,62)
        self.GlobalPool = nn.AdaptiveAvgPool1d(1)
        # (b,2048,1)
        #self.fc1 = nn.Linear(in_features=2048*1, out_features=128)
        #self.fc2 = nn.Linear(in_features=128, out_features=class_num)
        self.fc1 = KANLinear(in_features=2048*1, out_features=128)
        self.fc2 = KANLinear(in_features=128, out_features=class_num)
    def forward(self, x):
        output1 = self.PCSCB(x)
        output1 = self.conv1(output1)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output5 = self.conv5(output4)
        output6 = self.conv6(output5)
        output7 = self.GlobalPool(output6)
        output7 = output7.view(output7.size(0), -1)
        output7 = self.fc1(output7)
        output = self.fc2(output7)
        return F.log_softmax(output, dim=1)
########################################################################################################################
#  模型构建
# 1 经典网络结构
class CNN_2D(nn.Module):
    def __init__(self):
        super(CNN_2D, self).__init__()
        #121 80
        self.conv1 = Conv2dBlock(in_channels=4,out_channels=64,kernel_size=3,padding=1,stride=1,bn=1,pool_size=2,dropout=0)#60,40
        self.conv2 = Conv2dBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bn=1, pool_size=2,#30,20
                                 dropout=0)
        self.conv3 = Conv2dBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, bn=1, pool_size=2,#15,10
                                 dropout=0)
        self.conv4 = Conv2dBlock(in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=1, bn=1, pool_size=1,#13,8
                                 dropout=0)
        self.GlobalPool = nn.AdaptiveAvgPool2d(1)#512,1
        self.fc1 = nn.Linear(in_features=256*1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    def forward(self, x):
        #x1 = x[:, 0, :, :]
        #x2 = x[:, 1, :, :]
        #x=torch.stack((x1,x2), dim=1)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.GlobalPool(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        nn.Dropout(0.3)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DenseLayer(nn.Module):
    def __init__(self, input_c, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_c, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, inputs):
        output = self.bn1(inputs)
        output = self.relu1(output)
        output = self.conv1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv2(output)
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate)
        return output
class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, input_c, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(input_c + i * growth_rate,
                               growth_rate=growth_rate,
                               bn_size=bn_size,
                               drop_rate=drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            concat_features = torch.cat(features, 1)
            new_features = layer(concat_features)
            features.append(new_features)
        return torch.cat(features, 1)
class Transition(nn.Module):
    def __init__(self, input_c, output_c):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(input_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_c, output_c,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, input):
        output = self.bn(input)
        output = self.relu(output)
        output = self.conv(output)
        output = self.pool(output)
        return output
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                 compression_rate=0.5, drop_rate=0, num_classes=2):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers,
                               input_c=num_features,
                               bn_size=bn_size,
                               growth_rate=growth_rate,
                               drop_rate=drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = Transition(input_c=num_features, output_c=int(num_features * compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)
        self.tail = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        tail_output = self.tail(features)
        out = F.adaptive_avg_pool2d(tail_output, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
#######################################################
class CNN_1D(nn.Module):
    def __init__(self, class_num):
        super(CNN_1D, self).__init__()
        #(b,6,4000)
        self.conv1 = MSC1dBlock(in_channels=1,out_channels=128,bn=1,pool_size=2)
        #(b,128,2000)
        self.conv2 = MSC1dBlock(in_channels=128, out_channels=256, bn=1, pool_size=2)
        # (b,256,1000)
        self.conv3 = MSC1dBlock(in_channels=256, out_channels=512,bn=1, pool_size=2)
        # (b,256,500)
        self.conv4 = MSC1dBlock(in_channels=512, out_channels=1024, bn=1, pool_size=2)
        # (b,256,250)
        self.conv5 = MSC1dBlock(in_channels=1024, out_channels=2048, bn=1, pool_size=2)
        # (b,128,125)
        self.conv6 = MSC1dBlock(in_channels=2048, out_channels=2048, bn=1, pool_size=2)
        # (b,64,62)
        self.GlobalPool = nn.AdaptiveAvgPool1d(1)
        # (b,2048,1)
        self.fc1 = nn.Linear(in_features=2048*1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=class_num)
        self.relu = nn.ReLU()
    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output5 = self.conv5(output4)
        output6 = self.conv6(output5)
        output7 = self.GlobalPool(output6)
        output7 = output7.view(output7.size(0), -1)
        output7 = self.fc1(output7)
        output7 = self.relu(output7)
        output = self.fc2(output7)
        return F.log_softmax(output, dim=1)
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.hidden_dim = 1
        self.num_layers = 1

        self.lstm = nn.LSTM(1, 1, 1, batch_first=True)
        self.conv1 = MSC1dBlock(in_channels=1, out_channels=128, bn=1, pool_size=2)
        # (b,128,2000)
        self.conv2 = MSC1dBlock(in_channels=128, out_channels=256, bn=1, pool_size=2)
        # (b,256,1000)
        self.conv3 = MSC1dBlock(in_channels=256, out_channels=512, bn=1, pool_size=2)
        # (b,256,500)
        self.conv4 = MSC1dBlock(in_channels=512, out_channels=1024, bn=1, pool_size=2)
        # (b,256,250)
        self.conv5 = MSC1dBlock(in_channels=1024, out_channels=2048, bn=1, pool_size=2)
        # (b,128,125)
        self.conv6 = MSC1dBlock(in_channels=2048, out_channels=2048, bn=1, pool_size=2)
        # (b,64,62)
        self.GlobalPool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        self.lstm.flatten_parameters()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # 前向传播LSTM
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.GlobalPool(out)
        out, _ = self.lstm(out, (h0, c0))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
########################################################################################################################
def print_model_layers_parameters(model):
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential) and not isinstance(module,
                                                                                                      nn.ModuleList) and len(
                list(module.parameters())) > 0:
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: {params} parameters")
            total_params += params
    print(f"Total Trainable Params: {total_params}")
