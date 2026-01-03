import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import List, Tuple, Optional, Dict, Any

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 激活函数 ====================
class ReLU:
    """ReLU激活函数"""

    def __init__(self):
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.input = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        dx = dout.copy()
        dx[self.input <= 0] = 0
        return dx


class Softmax:
    """Softmax激活函数（数值稳定实现）"""

    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播（数值稳定）"""
        # 减去最大值以提高数值稳定性
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Softmax的反向传播通常在交叉熵损失中一起计算"""
        # 在交叉熵损失中处理，这里返回None
        return None


# ==================== 损失函数 ====================
class CrossEntropyLoss:
    """交叉熵损失函数（包含Softmax）"""

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """前向传播：计算交叉熵损失"""
        # y_pred: (batch_size, num_classes)
        # y_true: (batch_size, num_classes) one-hot编码

        # 数值稳定的softmax
        y_pred_max = np.max(y_pred, axis=1, keepdims=True)
        exp_y_pred = np.exp(y_pred - y_pred_max)
        y_pred_softmax = exp_y_pred / np.sum(exp_y_pred, axis=1, keepdims=True)

        # 添加小值防止log(0)
        y_pred_softmax = np.clip(y_pred_softmax, 1e-15, 1 - 1e-15)

        # 计算交叉熵损失
        batch_size = y_pred.shape[0]
        loss = -np.sum(y_true * np.log(y_pred_softmax)) / batch_size

        self.y_pred = y_pred_softmax
        self.y_true = y_true

        return loss

    def backward(self) -> np.ndarray:
        """反向传播：计算梯度"""
        if self.y_pred is None or self.y_true is None:
            raise ValueError("必须先调用forward方法")

        batch_size = self.y_true.shape[0]
        # Softmax + CrossEntropy的梯度公式: dL/dz = y_pred - y_true
        grad = (self.y_pred - self.y_true) / batch_size
        return grad


# ==================== 卷积层 ====================
class Conv2D:
    """2D卷积层"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, learning_rate: float = 0.01):
        """
        初始化卷积层

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（卷积核数量）
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小
            learning_rate: 学习率
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate

        # He初始化权重
        std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        self.bias = np.zeros((out_channels, 1))

        # 缓存用于反向传播
        self.input = None
        self.output = None
        self.grad_weights = None
        self.grad_bias = None

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """对输入进行填充"""
        if self.padding > 0:
            return np.pad(x, ((0, 0), (0, 0),
                              (self.padding, self.padding),
                              (self.padding, self.padding)),
                          mode='constant', constant_values=0)
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, in_channels, height, width)

        Returns:
            输出张量，形状为 (batch_size, out_channels, out_height, out_width)
        """
        self.input = x
        batch_size, in_channels, in_height, in_width = x.shape

        # 检查输入通道数
        if in_channels != self.in_channels:
            raise ValueError(f"输入通道数不匹配: 预期 {self.in_channels}, 得到 {in_channels}")

        # 应用填充
        x_padded = self._pad_input(x)

        # 计算输出尺寸
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 初始化输出
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # 执行卷积操作
        for i in range(out_height):
            for j in range(out_width):
                # 计算输入区域
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # 获取输入区域
                region = x_padded[:, :, h_start:h_end, w_start:w_end]

                # 卷积计算
                for k in range(self.out_channels):
                    # 卷积核与输入区域逐元素相乘并求和
                    output[:, k, i, j] = np.sum(region * self.weights[k, :, :, :], axis=(1, 2, 3))

                    # 添加偏置
                    output[:, k, i, j] += self.bias[k]

        self.output = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            dout: 上游梯度，形状为 (batch_size, out_channels, out_height, out_width)

        Returns:
            输入梯度，形状为 (batch_size, in_channels, height, width)
        """
        if self.input is None:
            raise ValueError("必须先调用forward方法")

        batch_size, out_channels, out_height, out_width = dout.shape
        _, in_channels, in_height, in_width = self.input.shape

        # 初始化梯度
        dx = np.zeros_like(self.input)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        # 如果需要填充，先填充输入
        input_padded = self._pad_input(self.input)
        dx_padded = np.zeros_like(input_padded) if self.padding > 0 else dx

        # 计算梯度
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # 获取输入区域
                region = input_padded[:, :, h_start:h_end, w_start:w_end]

                # 计算权重梯度
                for k in range(self.out_channels):
                    dout_slice = dout[:, k, i, j][:, np.newaxis, np.newaxis, np.newaxis]
                    self.grad_weights[k] += np.sum(dout_slice * region, axis=0)

                # 计算输入梯度
                for k in range(self.out_channels):
                    dx_padded[:, :, h_start:h_end, w_start:w_end] += \
                        dout[:, k, i, j][:, np.newaxis, np.newaxis, np.newaxis] * self.weights[k]

        # 如果使用了填充，需要去除填充
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded

        # 计算偏置梯度
        self.grad_bias = np.sum(dout, axis=(0, 2, 3)).reshape(-1, 1)

        return dx

    def update(self):
        """更新权重和偏置"""
        if self.grad_weights is None or self.grad_bias is None:
            raise ValueError("必须先调用backward方法")

        self.weights -= self.learning_rate * self.grad_weights
        self.bias -= self.learning_rate * self.grad_bias


# ==================== 最大池化层 ====================
class MaxPool2D:
    """2D最大池化层"""

    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        初始化最大池化层

        Args:
            pool_size: 池化窗口大小
            stride: 步长
        """
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, channels, height, width)

        Returns:
            输出张量，形状为 (batch_size, channels, out_height, out_width)
        """
        self.input = x
        batch_size, channels, in_height, in_width = x.shape

        # 计算输出尺寸
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1

        # 初始化输出和掩码
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.mask = np.zeros_like(x)

        # 执行最大池化
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # 获取池化区域
                region = x[:, :, h_start:h_end, w_start:w_end]

                # 计算最大值
                output[:, :, i, j] = np.max(region, axis=(2, 3))

                # 创建掩码（用于反向传播）
                region_reshaped = region.reshape(batch_size, channels, -1)
                max_indices = np.argmax(region_reshaped, axis=2)

                # 将最大值位置标记为1
                for b in range(batch_size):
                    for c in range(channels):
                        idx = max_indices[b, c]
                        h_idx = idx // self.pool_size
                        w_idx = idx % self.pool_size
                        self.mask[b, c, h_start + h_idx, w_start + w_idx] = 1

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            dout: 上游梯度，形状为 (batch_size, channels, out_height, out_width)

        Returns:
            输入梯度，形状为 (batch_size, channels, height, width)
        """
        if self.input is None or self.mask is None:
            raise ValueError("必须先调用forward方法")

        batch_size, channels, in_height, in_width = self.input.shape
        _, _, out_height, out_width = dout.shape

        # 初始化梯度
        dx = np.zeros_like(self.input)

        # 根据掩码将梯度传递回最大值位置
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # 将梯度放回最大值位置
                mask_region = self.mask[:, :, h_start:h_end, w_start:w_end]
                dx[:, :, h_start:h_end, w_start:w_end] += \
                    dout[:, :, i, j][:, :, np.newaxis, np.newaxis] * mask_region

        return dx

    def update(self):
        """池化层没有可训练参数"""
        pass


# ==================== 全连接层 ====================
class Dense:
    """全连接层"""

    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 0.01):
        """
        初始化全连接层

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            learning_rate: 学习率
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # He初始化权重
        std = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(output_dim, input_dim) * std
        self.bias = np.zeros((output_dim, 1))

        # 缓存
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, input_dim)

        Returns:
            输出张量，形状为 (batch_size, output_dim)
        """
        # 如果输入是4D的（来自卷积层），展平它
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        self.input = x
        # 全连接计算: y = xW^T + b
        output = np.dot(x, self.weights.T) + self.bias.T
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            dout: 上游梯度，形状为 (batch_size, output_dim)

        Returns:
            输入梯度，形状为 (batch_size, input_dim)
        """
        if self.input is None:
            raise ValueError("必须先调用forward方法")

        batch_size = dout.shape[0]

        # 计算梯度
        self.grad_weights = np.dot(dout.T, self.input) / batch_size
        self.grad_bias = np.sum(dout, axis=0, keepdims=True).T / batch_size

        # 计算输入梯度
        dx = np.dot(dout, self.weights)

        return dx

    def update(self):
        """更新权重和偏置"""
        if self.grad_weights is None or self.grad_bias is None:
            raise ValueError("必须先调用backward方法")

        self.weights -= self.learning_rate * self.grad_weights
        self.bias -= self.learning_rate * self.grad_bias


# ==================== Flatten层 ====================
class Flatten:
    """展平层（将多维输入展平为一维）"""

    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        return dout.reshape(self.input_shape)

    def update(self):
        """没有可训练参数"""
        pass


# ==================== CNN模型类 ====================
class CNN:
    """卷积神经网络模型"""

    def __init__(self, learning_rate: float = 0.01):
        """
        初始化CNN模型

        Args:
            learning_rate: 学习率
        """
        self.layers = []
        self.learning_rate = learning_rate
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def add(self, layer):
        """添加层到模型"""
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入数据

        Returns:
            模型输出
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray):
        """
        反向传播

        Args:
            grad: 损失函数的梯度
        """
        # 反向传播通过所有层
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self):
        """更新所有层的权重"""
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update()

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 10, batch_size: int = 32,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            X_train: 训练数据
            y_train: 训练标签（one-hot编码）
            X_val: 验证数据
            y_val: 验证标签（one-hot编码）
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否显示训练过程

        Returns:
            训练历史记录
        """
        num_train = X_train.shape[0]
        num_batches = int(np.ceil(num_train / batch_size))

        # 初始化损失函数
        criterion = CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # 打乱训练数据
            indices = np.random.permutation(num_train)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0
            correct = 0
            total = 0

            # 批量训练
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_train)

                # 获取当前批次数据
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # 前向传播
                output = self.forward(X_batch)

                # 计算损失
                loss = criterion.forward(output, y_batch)
                epoch_loss += loss * (end - start)

                # 计算准确率
                predictions = np.argmax(output, axis=1)
                labels = np.argmax(y_batch, axis=1)
                correct += np.sum(predictions == labels)
                total += (end - start)

                # 反向传播
                grad = criterion.backward()
                self.backward(grad)

                # 更新权重
                self.update()

                # 显示批量训练进度
                if verbose and (batch_idx + 1) % 100 == 0:
                    print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.4f}")

            # 计算训练集损失和准确率
            train_loss = epoch_loss / num_train
            train_acc = correct / total
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            # 验证集评估
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)

            epoch_time = time.time() - epoch_start_time

            # 显示训练结果
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"Time: {epoch_time:.2f}s")

        # 返回训练历史
        history = {
            'train_loss': self.train_loss_history,
            'train_acc': self.train_acc_history,
            'val_loss': self.val_loss_history,
            'val_acc': self.val_acc_history
        }

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[float, float]:
        """
        评估模型

        Args:
            X: 输入数据
            y: 标签（one-hot编码）
            batch_size: 批次大小

        Returns:
            损失值和准确率
        """
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        criterion = CrossEntropyLoss()
        total_loss = 0
        correct = 0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)

            X_batch = X[start:end]
            y_batch = y[start:end]

            # 前向传播
            output = self.forward(X_batch)

            # 计算损失
            loss = criterion.forward(output, y_batch)
            total_loss += loss * (end - start)

            # 计算准确率
            predictions = np.argmax(output, axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += np.sum(predictions == labels)

        avg_loss = total_loss / num_samples
        accuracy = correct / num_samples

        return avg_loss, accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入数据

        Returns:
            预测结果
        """
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions


# ==================== 数据集处理 ====================
def load_mnist_data(reduce_data: bool = False, reduce_factor: float = 0.1) -> Tuple:
    """
    加载MNIST数据集（从本地mnist.npz文件加载）

    Args:
        reduce_data: 是否减少数据量（用于快速测试）
        reduce_factor: 数据减少因子

    Returns:
        (X_train, y_train, X_test, y_test) 训练和测试数据
    """
    print("正在加载MNIST数据集...")
    start_time = time.time()

    # 首先尝试从本地mnist.npz文件加载
    local_mnist_path = './mnist.npz'

    if os.path.exists(local_mnist_path):
        print(f"从本地文件加载: {local_mnist_path}")
        try:
            with np.load(local_mnist_path, allow_pickle=True) as f:
                X_train = f['x_train']
                y_train = f['y_train']
                X_test = f['x_test']
                y_test = f['y_test']

            print(f"成功加载本地MNIST数据")
            print(f"原始数据形状 - X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"原始数据形状 - X_test: {X_test.shape}, y_test: {y_test.shape}")

        except Exception as e:
            print(f"从本地文件加载失败: {e}")
            print("将尝试备用方案...")
            # 如果本地文件加载失败，尝试从网络加载
            return load_mnist_data_from_web(reduce_data, reduce_factor)
    else:
        print(f"本地文件 {local_mnist_path} 不存在")
        print("将尝试从网络加载...")
        return load_mnist_data_from_web(reduce_data, reduce_factor)

    # 归一化到[0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 添加通道维度 (N, H, W) -> (N, H, W, C)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # 转换为(N, C, H, W)格式（我们的卷积层使用这种格式）
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # 标签转换为one-hot编码
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]

    # 如果需要减少数据量（用于快速测试）
    if reduce_data:
        train_samples = int(X_train.shape[0] * reduce_factor)
        test_samples = int(X_test.shape[0] * reduce_factor)

        X_train = X_train[:train_samples]
        y_train = y_train[:train_samples]
        y_train_onehot = y_train_onehot[:train_samples]

        X_test = X_test[:test_samples]
        y_test = y_test[:test_samples]
        y_test_onehot = y_test_onehot[:test_samples]

        print(f"减少数据量: 训练集 {train_samples} 样本, 测试集 {test_samples} 样本")

    print(f"数据集加载完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"处理后数据形状 - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"处理后数据形状 - X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


def load_mnist_data_from_web(reduce_data: bool = False, reduce_factor: float = 0.1) -> Tuple:
    """
    从网络加载MNIST数据集（备用方案）

    Args:
        reduce_data: 是否减少数据量（用于快速测试）
        reduce_factor: 数据减少因子

    Returns:
        (X_train, y_train, X_test, y_test) 训练和测试数据
    """
    print("正在从网络加载MNIST数据集...")

    try:
        # 使用sklearn的fetch_openml作为备用方案
        from sklearn.datasets import fetch_openml
        print("使用sklearn的fetch_openml加载MNIST数据集...")

        # 增加超时设置
        import urllib
        original_urlopen = urllib.request.urlopen
        urllib.request.urlopen = lambda url, timeout=30: original_urlopen(url, timeout=timeout)

        mnist_data = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
        X, y = mnist_data['data'], mnist_data['target'].astype(int)

        # 分割训练集和测试集
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        # 重塑为28x28图像
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)

        print("网络数据加载成功")

    except Exception as e:
        print(f"网络加载失败: {e}")
        print("正在生成模拟数据用于测试...")
        # 生成模拟数据
        X_train = np.random.randn(6000, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 6000)
        X_test = np.random.randn(1000, 28, 28).astype(np.float32)
        y_test = np.random.randint(0, 10, 1000)

    # 归一化到[0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 添加通道维度 (N, H, W) -> (N, H, W, C)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # 转换为(N, C, H, W)格式（我们的卷积层使用这种格式）
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # 标签转换为one-hot编码
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]

    # 如果需要减少数据量（用于快速测试）
    if reduce_data:
        train_samples = int(X_train.shape[0] * reduce_factor)
        test_samples = int(X_test.shape[0] * reduce_factor)

        X_train = X_train[:train_samples]
        y_train = y_train[:train_samples]
        y_train_onehot = y_train_onehot[:train_samples]

        X_test = X_test[:test_samples]
        y_test = y_test[:test_samples]
        y_test_onehot = y_test_onehot[:test_samples]

        print(f"减少数据量: 训练集 {train_samples} 样本, 测试集 {test_samples} 样本")

    print(f"数据集加载完成")
    print(f"处理后数据形状 - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"处理后数据形状 - X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


# ==================== 可视化函数 ====================
def plot_training_history(history: Dict[str, List[float]], save_dir: str = None, show: bool = True):
    """
    绘制训练历史曲线

    Args:
        history: 训练历史记录
        save_dir: 保存目录路径（如果为None则不保存）
        show: 是否显示图形
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('损失', fontsize=12)
    axes[0].set_title('训练和验证损失', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=10)

    # 绘制准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('准确率', fontsize=12)
    axes[1].set_title('训练和验证准确率', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=10)

    plt.tight_layout()

    # 保存图形
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'training_history_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"训练历史图形已保存到: {filename}")

        # 同时保存为PDF格式（矢量图，适合印刷）
        pdf_filename = os.path.join(save_dir, f'training_history_{timestamp}.pdf')
        plt.savefig(pdf_filename, bbox_inches='tight')
        print(f"训练历史图形(矢量)已保存到: {pdf_filename}")

    # 显示图形
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_predictions(model: CNN, X_test: np.ndarray, y_test: np.ndarray,
                          num_samples: int = 10, save_dir: str = None, show: bool = True):
    """
    可视化测试样本的预测结果

    Args:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签
        num_samples: 要可视化的样本数量
        save_dir: 保存目录路径（如果为None则不保存）
        show: 是否显示图形
    """
    # 随机选择样本
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    samples = X_test[indices]
    true_labels = y_test[indices]

    # 进行预测
    predictions = model.predict(samples)

    # 创建可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        # 将图像转回(H, W)格式用于显示
        img = samples[i].transpose(1, 2, 0).squeeze()

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

        # 设置标题颜色：绿色表示正确，红色表示错误
        if predictions[i] == true_labels[i]:
            color = 'green'
            result = '正确'
        else:
            color = 'red'
            result = '错误'

        axes[i].set_title(f'真实: {true_labels[i]}\n预测: {predictions[i]}\n({result})',
                          color=color, fontsize=10)

    plt.suptitle('测试样本预测结果', fontsize=14, y=1.02)
    plt.tight_layout()

    # 保存图形
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'predictions_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"预测结果图形已保存到: {filename}")

        # 同时保存为PDF格式
        pdf_filename = os.path.join(save_dir, f'predictions_{timestamp}.pdf')
        plt.savefig(pdf_filename, bbox_inches='tight')
        print(f"预测结果图形(矢量)已保存到: {pdf_filename}")

    # 显示图形
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 打印准确率统计
    correct = np.sum(predictions == true_labels)
    accuracy = correct / num_samples * 100
    print(f"随机 {num_samples} 个样本的准确率: {accuracy:.1f}% ({correct}/{num_samples})")

    return accuracy, correct


def save_training_results_to_txt(model: CNN, history: Dict[str, List[float]],
                                 test_accuracy: float, test_loss: float,
                                 training_params: Dict[str, Any], save_dir: str = None):
    """
    将训练结果保存到文本文件

    Args:
        model: 训练好的模型
        history: 训练历史记录
        test_accuracy: 测试集准确率
        test_loss: 测试集损失
        training_params: 训练参数
        save_dir: 保存目录路径（如果为None则不保存）
    """
    if not save_dir:
        return

    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f'training_results_{timestamp}.txt')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CNN MNIST手写数字分类 - 训练结果报告\n")
        f.write("=" * 60 + "\n\n")

        # 写入训练参数
        f.write("训练参数:\n")
        f.write("-" * 40 + "\n")
        for key, value in training_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # 写入模型架构
        f.write("模型架构:\n")
        f.write("-" * 40 + "\n")
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            f.write(f"第{i + 1}层: {layer_type}\n")

            # 添加特定层的详细信息
            if layer_type == "Conv2D":
                f.write(f"    输入通道: {layer.in_channels}, 输出通道: {layer.out_channels}, "
                        f"卷积核大小: {layer.kernel_size}\n")
            elif layer_type == "Dense":
                f.write(f"    输入维度: {layer.input_dim}, 输出维度: {layer.output_dim}\n")

        f.write("\n")

        # 写入训练历史
        f.write("训练历史:\n")
        f.write("-" * 40 + "\n")
        epochs = len(history['train_loss'])
        for epoch in range(epochs):
            f.write(f"Epoch {epoch + 1}: ")
            f.write(f"训练损失={history['train_loss'][epoch]:.4f}, ")
            f.write(f"训练准确率={history['train_acc'][epoch]:.4f}, ")
            f.write(f"验证损失={history['val_loss'][epoch]:.4f}, ")
            f.write(f"验证准确率={history['val_acc'][epoch]:.4f}\n")

        f.write("\n")

        # 写入最终测试结果
        f.write("最终测试结果:\n")
        f.write("-" * 40 + "\n")
        f.write(f"测试集损失: {test_loss:.4f}\n")
        f.write(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n")

        # 写入时间戳
        f.write("\n")
        f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")

    print(f"训练结果报告已保存到: {filename}")


def save_confusion_matrix(model: CNN, X_test: np.ndarray, y_test: np.ndarray,
                          save_dir: str = None, show: bool = True):
    """
    生成并保存混淆矩阵

    Args:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签
        save_dir: 保存目录路径（如果为None则不保存）
        show: 是否显示图形
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # 进行预测
    predictions = model.predict(X_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵', fontsize=14)
    plt.tight_layout()

    # 保存图形
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {filename}")

        # 同时保存为PDF格式
        pdf_filename = os.path.join(save_dir, f'confusion_matrix_{timestamp}.pdf')
        plt.savefig(pdf_filename, bbox_inches='tight')
        print(f"混淆矩阵(矢量)已保存到: {pdf_filename}")

    # 显示图形
    if show:
        plt.show()
    else:
        plt.close()

    # 计算并打印每个类别的准确率
    class_accuracy = []
    for i in range(10):
        correct = np.sum((y_test == i) & (predictions == i))
        total = np.sum(y_test == i)
        accuracy = correct / total if total > 0 else 0
        class_accuracy.append(accuracy)
        print(f"数字 {i} 的准确率: {accuracy:.4f} ({correct}/{total})")

    return cm, class_accuracy


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("NumPy CNN MNIST手写数字分类")
    print("=" * 60)

    # 参数设置
    REDUCE_DATA = True  # 是否减少数据量用于快速测试
    REDUCE_FACTOR = 0.1  # 数据减少因子
    EPOCHS = 5  # 训练轮数
    BATCH_SIZE = 32  # 批次大小
    LEARNING_RATE = 0.01  # 学习率

    # 设置保存目录
    SAVE_DIR = "training_results"
    SAVE_IMAGES = True  # 是否保存图形

    # 创建保存目录
    if SAVE_IMAGES:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"结果将保存到目录: {SAVE_DIR}")

    # 检查本地mnist.npz文件是否存在
    if not os.path.exists('./mnist.npz'):
        print("警告: 当前目录下未找到 mnist.npz 文件")
        print("请确保 mnist.npz 文件位于当前目录")
        print("您可以从以下链接下载: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
        response = input("是否继续使用网络加载? (y/n): ")
        if response.lower() != 'y':
            print("程序退出")
            return

    # 1. 加载MNIST数据集
    print("\n1. 加载数据集...")
    X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot = load_mnist_data(
        reduce_data=REDUCE_DATA, reduce_factor=REDUCE_FACTOR
    )

    # 2. 创建CNN模型
    print("\n2. 创建CNN模型...")
    model = CNN(learning_rate=LEARNING_RATE)

    # 构建CNN架构
    # 输入: (batch_size, 1, 28, 28)
    model.add(Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=1, learning_rate=LEARNING_RATE))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))  # 输出: (batch_size, 16, 14, 14)

    model.add(Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1, learning_rate=LEARNING_RATE))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))  # 输出: (batch_size, 32, 7, 7)

    model.add(Flatten())  # 输出: (batch_size, 32*7*7 = 1568)

    model.add(Dense(input_dim=1568, output_dim=128, learning_rate=LEARNING_RATE))
    model.add(ReLU())

    model.add(Dense(input_dim=128, output_dim=10, learning_rate=LEARNING_RATE))
    # 注意: Softmax在交叉熵损失函数中实现

    print("模型架构创建完成")

    # 3. 训练模型
    print("\n3. 训练模型...")
    print(f"训练参数: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")

    start_time = time.time()
    history = model.train(
        X_train, y_train_onehot,
        X_test, y_test_onehot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    training_time = time.time() - start_time

    print(f"\n训练完成! 总训练时间: {training_time:.2f}秒")
    print(f"平均每轮训练时间: {training_time / EPOCHS:.2f}秒")

    # 4. 评估模型
    print("\n4. 评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, batch_size=BATCH_SIZE)
    print(f"测试集 - 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")

    # 5. 可视化训练过程并保存
    print("\n5. 可视化训练过程...")
    plot_training_history(history, save_dir=SAVE_DIR if SAVE_IMAGES else None, show=True)

    # 6. 可视化预测结果并保存
    print("\n6. 可视化预测结果...")
    sample_accuracy, correct_samples = visualize_predictions(
        model, X_test, y_test, num_samples=10,
        save_dir=SAVE_DIR if SAVE_IMAGES else None, show=True
    )

    # 7. 生成并保存混淆矩阵
    print("\n7. 生成混淆矩阵...")
    cm, class_accuracies = save_confusion_matrix(
        model, X_test, y_test,
        save_dir=SAVE_DIR if SAVE_IMAGES else None, show=True
    )

    # 8. 保存训练结果到文本文件
    print("\n8. 保存训练结果报告...")
    training_params = {
        "Epochs": EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Learning Rate": LEARNING_RATE,
        "Reduce Data": REDUCE_DATA,
        "Reduce Factor": REDUCE_FACTOR,
        "Training Time (s)": f"{training_time:.2f}",
        "Average Time per Epoch (s)": f"{training_time / EPOCHS:.2f}",
        "Test Accuracy": f"{test_acc:.4f}",
        "Test Loss": f"{test_loss:.4f}"
    }

    save_training_results_to_txt(
        model, history, test_acc, test_loss, training_params,
        save_dir=SAVE_DIR if SAVE_IMAGES else None
    )

    # 9. 保存模型权重
    print("\n9. 保存模型权重...")
    try:
        def save_model_weights(model, save_dir=None, filename='cnn_weights.npz'):
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, filename)
            else:
                filepath = filename

            weights = {}
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'weights'):
                    weights[f'layer_{i}_weights'] = layer.weights
                if hasattr(layer, 'bias'):
                    weights[f'layer_{i}_bias'] = layer.bias
            np.savez(filepath, **weights)
            print(f"模型权重已保存到 {filepath}")

            # 同时保存模型架构信息
            arch_filename = filepath.replace('.npz', '_architecture.txt')
            with open(arch_filename, 'w', encoding='utf-8') as f:
                f.write("模型架构信息:\n")
                f.write("=" * 40 + "\n")
                for i, layer in enumerate(model.layers):
                    f.write(f"第{i + 1}层: {type(layer).__name__}\n")
                    if hasattr(layer, 'input_dim'):
                        f.write(f"    输入维度: {layer.input_dim}\n")
                    if hasattr(layer, 'output_dim'):
                        f.write(f"    输出维度: {layer.output_dim}\n")
                    if hasattr(layer, 'in_channels'):
                        f.write(f"    输入通道: {layer.in_channels}\n")
                    if hasattr(layer, 'out_channels'):
                        f.write(f"    输出通道: {layer.out_channels}\n")
                    if hasattr(layer, 'kernel_size'):
                        f.write(f"    卷积核大小: {layer.kernel_size}\n")
                f.write(f"\n保存时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"模型架构信息已保存到 {arch_filename}")

        save_model_weights(model, save_dir=SAVE_DIR if SAVE_IMAGES else None)
    except Exception as e:
        print(f"保存模型权重时出错: {e}")

    # 10. 创建结果摘要
    print("\n" + "=" * 60)
    print("训练结果摘要:")
    print("-" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
    print(f"最终测试损失: {test_loss:.4f}")
    print(f"总训练时间: {training_time:.2f}秒")
    print(f"每轮平均时间: {training_time / EPOCHS:.2f}秒")

    if SAVE_IMAGES:
        print(f"\n所有结果已保存到目录: {os.path.abspath(SAVE_DIR)}")
        print(f"包含以下文件:")
        print(f"  - training_history_*.png/pdf: 训练历史曲线")
        print(f"  - predictions_*.png/pdf: 预测结果可视化")
        print(f"  - confusion_matrix_*.png/pdf: 混淆矩阵")
        print(f"  - training_results_*.txt: 训练结果报告")
        print(f"  - cnn_weights.npz: 模型权重")
        print(f"  - cnn_weights_architecture.txt: 模型架构信息")

    print("\n" + "=" * 60)
    print("程序执行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()