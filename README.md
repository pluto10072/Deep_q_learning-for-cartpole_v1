这是一个基于深度double-q-learning的cartpole_v1参考解决方案，有很多不足之处

方案效果：训练了200个episode最高reward达到459，取该策略网络参数测试10个episode，10次测试平均reward: 204.4，最小Reward不低于180.满足解决条件

本项目采用如下的全连接神经网络作为 Q 网络（policy_net 和 target_net）：

输入层：4 个神经元（对应 CartPole 的状态空间）

隐藏层1：32 个神经元，ReLU 激活

隐藏层2：64 个神经元，ReLU 激活

输出层：2 个神经元（对应动作空间，分别为向左和向右推杆）


主要超参数说明：
learning_rate = 0.001

gamma = 0.9

epsilon_start = 0.5

epsilon_decay：（epsilon -= epsilon_start/episodes）

batch_size = 32

buffer_size = 1,000,000

target_update = 20

Ps：在代码中我添加了对于杆位置与角度的惩罚，初步做了线性的惩罚叠加奖励，但是在训练的过程中收敛的效果很差，所以在代码中注释掉了
