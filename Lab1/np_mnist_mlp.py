# -*- coding: utf-8 -*-
"""
@ author: xusinga@stu.pku.edu.cn
"""

# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm  import tqdm


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码, for number 0~9, [y_train]作为索引

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    return np.maximum(0, x)

def relu_prime(x:np.ndarray):
    return np.where(x > 0, 1, 0)

def leakyReLU(x,alpha = 0.01):
    return np.where(x > 0, x, alpha*x)
    
def leakyReLU_prime(x,alpha = 0.01):
    return np.where(x >= 0, 1, alpha)

def stable_sigmoid(x):
    # 需要对x分情况处理，避免计算大指数导致上溢问题
    # 可以证明x < 0时改为计算 e^x / (1 + e^x)是数学等价的（上下同乘以e^{-x}）
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1+np.exp(x)))

def sigmoid_prime(x):
    s = stable_sigmoid(x)
    return s * (1 - s)

def stable_tanh(x):    
    # 通过数学变形和阈值截断避免上下溢出问题和优化运算花费
    threshold = 20 # 设置阈值，避免计算大指数
        
    return np.where(x >= 0, 
                   np.where(x > threshold, 1, (1 - np.exp(-2*x)) / (1 +  np.exp(-2*x))), # 数学变形
                   np.where(np.abs(x) > threshold,-1, -(1 - np.exp(2*x)) / (1 +  np.exp(2*x))) # 利用奇函数性质
                )

def tanh_prime(x):
    return 1-stable_tanh(x)**2

# 选择激活函数
def activate_func(x:np.ndarray)->np.ndarray:
    """激活函数

    Args:
        x (np.ndarray): input_data

    Returns:
        f(x) (np.ndarray): same shape as x, activated each element in x
    """
    return relu(x)

def activate_func_prime(x:np.ndarray)->np.ndarray:
    """激活函数的导数

    Args:
        x (np.ndarray): input_data

    Returns:
        f(x) (np.ndarray): same shape as x, each element is the derivative of the activation function applied to the corresponding element in x
    """
    return relu_prime(x)

#输出层激活函数
def softmax(x:np.ndarray):
    """softmax函数, 防止除0

    Args:
        x (np.ndarray): in shape (batch_size, feature_size) or (feature_size)

    Returns:
        s (np.ndarray): same shape with x; 分别对x各行softmax后的结果; 
    """
    stable_expx = np.exp(x - np.max(x,axis=-1,keepdims=True)) # 减去最大值，解决上溢问题和缓解下溢问题（利用softmax函数的平移不变性和指数运算特性）
    dividor = np.sum(stable_expx,axis=-1,keepdims=True) # shape:(batch_size,1) or (1); exp后各行的sum
    result = stable_expx/dividor # dividor沿行方向广播，把第一列扩展填充至匹配stable_expx
    return result + 1e-8 # 添加平滑项,避免下溢问题

def softmax_prime(x):
    """softmax函数的导数
    Args:
        x (np.ndarray): in shape (feature_size)
    Returns:
        A (np.ndarray): in shape (feature_size, feature_size), A(i;j) = partial s_i on x_j
    """
    s = softmax(x)
    return np.diag(s) - np.outer(s, s)

# 定义损失函数
def loss_fn(y_true:np.ndarray, y_pred:np.ndarray):
    """Cross-entropy Loss

    Args:
        y_true (np.ndarray): (batch_size, num_classes), one-hot编码
        y_pred (np.ndarray): (batch_size, num_classes), softmax输出

    Returns:
        loss (np.ndarray): (batch_size), the loss of each batch
    """
    loss = -np.sum(y_true*np.log(y_pred), axis=-1)
    return loss

def loss_fn_prime(y_true, y_pred):
    """prime of Cross-Entropy Loss combine with softmax

    Args:
        y_true (np.ndarray): (batch_size, num_classes), one-hot编码
        y_pred (np.ndarray): (batch_size, num_classes), softmax输出

    Returns:
        partialLossOnZ (np.ndarray): (batch_size, num_classes), partial loss on partial z_i
    """
    # (partial loss on z_i) = Sum_j (partial loss on y_pred_j)*(partial y_pred_j on z_i) = y_pred_i - y_true_i
    return y_pred-y_true 
    

# 定义权重初始化函数
def init_weights(shape=()):
    '''
    He初始化权重（ReLu下最优）
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, hidden_layer_cnt=1):
        '''
        初始化网络结构
        '''
        self.weight_of_layers = [] # 元素为对应层的权重矩阵
        self.bias_of_layers = [] # 元素为对应层的偏置向量
        self.hidden_layer_cnt = hidden_layer_cnt # 隐藏层数量（即不计算softmax输出层）
        
        # 初始化第i层到第i+1层的权重和偏置（包括从L-1层到L层的）
        for i in range(self.hidden_layer_cnt+1):
            self.weight_of_layers.append(    
                init_weights(
                    (input_size if i == 0 else hidden_size, 
                     output_size if i == self.hidden_layer_cnt else hidden_size)
                )
            )
            self.bias_of_layers.append(
                np.zeros(shape=(1, output_size if i == self.hidden_layer_cnt else hidden_size))
            )
            
        self.Z = [0]*(self.hidden_layer_cnt+1) # list of 2D ndarrays(batch_size, hidden_size), 对应层的线性输出z
        self.A = [0]*(self.hidden_layer_cnt+1) # list of 2D ndarrays(batch_size, hidden_size), 对应层的z的激活结果
        self.w_grad_of_layers = [0]*(self.hidden_layer_cnt+1) # 逆序(索引0为第L层)，元素为对应层的w对本次batch的Loss的梯度平均
        self.b_grad_of_layers = [0]*(self.hidden_layer_cnt+1) # 逆序(索引0为第L层)，元素为对应层的b对本次batch的Loss的梯度平均
        self.lr = lr # 学习率

    def forward(self, x_batch):
        """前向传播

        Args:
            x_batch (np.ndarray): (batch_size,feature_size)

        Returns:
            a^L (np.ndarray): (batch_size, output_size) the output layer
        """
        for i in range(self.hidden_layer_cnt+1):
            if i == 0:
                self.Z[i] = x_batch @ self.weight_of_layers[i] + self.bias_of_layers[i] #z^1
            else:
                self.Z[i] = self.A[i-1] @ self.weight_of_layers[i] + self.bias_of_layers[i] #z^l
            if i == self.hidden_layer_cnt:
                self.A[i] = softmax(self.Z[i]) #a^L
            else:
                self.A[i] = activate_func(self.Z[i]) #a^l
        
        return self.A[-1] #a^L
    
    def backward(self,x_batch,y_batch):
        """反向传播，计算各层梯度

        Args:
            x_batch (np.ndarray): (batch_size,feature_size), batch input
            y_batch (np.ndarray): (batch_size,num_classes), batch ground truth
        """
        batch_size = x_batch.shape[0]
        prev_delta = pL_pZL = loss_fn_prime(y_batch,self.A[-1])
        for l in range(self.hidden_layer_cnt,-1,-1):
            if l == self.hidden_layer_cnt:
                pL_pZl = pL_pZL 
            else:
                pL_pZl =  prev_delta @ self.weight_of_layers[l+1].T * relu_prime(self.Z[l])
            pZl_pWl = self.A[l-1] if l>0 else x_batch
            pZl_pBl = np.ones(shape=(batch_size,1))
            self.w_grad_of_layers[l] = pZl_pWl.T @ pL_pZl / batch_size
            self.b_grad_of_layers[l] = pZl_pBl.T @ pL_pZl / batch_size
            prev_delta = pL_pZl
    
    
    def step(self, x_batch, y_batch):
        '''
        一步训练，并返回批平均损失和批预测准确率
        '''
        batch_size = x_batch.shape[0]        
        
        # 前向传播
        net_batch_output = self.forward(x_batch)
        
        # 计算损失和准确率
        avg_batch_loss = np.mean(loss_fn(y_batch, net_batch_output))
        batch_acc  = np.mean(np.argmax(net_batch_output, axis=1) == np.argmax(y_batch, axis=1)) # 等号左右的argmax返回一维ndarray, ndarray布尔表达式返回一维布尔数组
        
        # 反向传播
        self.backward(x_batch,y_batch)
        
        # 更新权重
        for i in range(self.hidden_layer_cnt+1):

            self.weight_of_layers[i] -=  self.w_grad_of_layers[i] * self.lr
            self.bias_of_layers[i] -=  self.b_grad_of_layers[i] * self.lr

        return avg_batch_loss, batch_acc


if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.01, hidden_layer_cnt=2)
    batch_size = 64
    for epoch in range(10):
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), batch_size))
        for i in p_bar:
            x_batch = X_train[i:min(i+batch_size,len(X_train))]
            y_batch = y_train[i:min(i+batch_size,len(y_train))]
            loss,acc = net.step(x_batch,y_batch)
            losses.append(loss)
            accuracies.append(acc)
            p_bar.set_description(f"epoch {epoch+1} training")

        # validation set
        val_y_pred = net.forward(X_val)
        val_loss = np.mean(loss_fn(y_val, val_y_pred))
        val_acc = np.mean(np.argmax(val_y_pred, axis=-1) == np.argmax(y_val, axis=-1))

        # 打印本epoch训练结果
        p_bar.write(f"epoch {epoch+1} training result:")
        p_bar.write(f"\ttrain_loss (avg): {np.mean(losses):.4f}")
        p_bar.write(f"\ttrain_acc (avg): {np.mean(accuracies)*100:.2f}%")
        p_bar.write(f"\tval_loss: {val_loss:.4f}")
        p_bar.write(f"\tval_acc: {val_acc*100:.2f}%")
        p_bar.write("")
        
        losses.clear()
        accuracies.clear()

    # test set
    test_y_pred = net.forward(X_test)
    test_loss = np.mean(loss_fn(y_test, test_y_pred))
    test_acc = np.mean(np.argmax(test_y_pred, axis=-1) == np.argmax(y_test, axis=-1))
    p_bar.write(f"Final test: \n\ttest_loss: {test_loss:.4f}\n\ttest_acc: {test_acc*100:.2f}%")
