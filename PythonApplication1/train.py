#训练神经网络，将参数(weight)存入HDF5文件
import numpy as np
import tensorflow as tf 
from network import *
from utils import *

def train():
    notes=get_notes()
    #得到所有不重复的音调数目
    num_pitch=len(set(notes))
    network_input,network_output=prepare_sequences(notes,num_pitch)
    model=network_model(network_input,num_pitch)
    #输入，音符的数量，训练后的参数文件(训练的时候不用写)
    filepath="weights-{epoch:02d}-{loss:.4f}.hdf5"
    
    #用checkpoint(检查点)文件在每一个Epoch结束时保存模型的参数
    #不怕训练过程中丢失模型参数，当对loss损失满意的时候可以随时停止训练
    checkpoint=tf.keras.callbacks.ModelCheckpoint(
        filepath,#保存参数文件的路径
        monitor='loss',#衡量的标准
        verbose=0,#不用冗余模式
        save_best_only=True,#最近出现的用monitor衡量的最好的参数不会被覆盖
        mode='min'#关注的是loss的最小值
    )
    #callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    #用fit方法来训练模型
    model.fit(network_input,network_output,epochs=90,batch_size=64,callbacks=callbacks_list)
    #输入，标签（衡量预测结果的），轮数，一次迭代的样本数，回调
    #model.save(filepath='./model',save_format='h5')

def prepare_sequences(notes,num_pitch):
    #从midi中读取的notes和所有音符的数量
    """ 
    为神经网络提供好要训练的序列 
    """
    sequence_length=100#序列长度

    #得到所有不同音高的名字
    pitch_names=sorted(set(item for item in notes))
    #把notes中的所有音符做集合操作，去掉重复的音，然后按照字母顺序排列

    #创建一个字典，用于映射 音高 和 整数
    pitch_to_int=dict((pitch,num)for num,pitch in enumerate(pitch_names))
    #枚举到pitch_name中

    #创建神经网络的输入序列和输出序列
    network_input=[]
    network_output=[]
    for i in range(0,len(notes)-sequence_length,1):#循环次数，步长为1
        sequence_in=notes[i:i+sequence_length]
        #每次输入100个序列，每隔长度1取下一组，例如：(0,100),(1,101),(50,150)
        sequence_out=notes[i+sequence_length]
        #真实值，从100开始往后
        network_input.append([pitch_to_int[char] for char in sequence_in])#列表生成式
        #把sequence_in中的每个字符转为整数（pitch_to_int[char]）放到network_input
        network_output.append(pitch_to_int[sequence_out])
        #把sequence_out的一个字符转为整数
    
    n_patterns=len(network_input)#输入序列长度
    
    #将输入序列的形状转成神经网络模型可以接受的
    network_input=np.reshape(network_input,(n_patterns,sequence_length,1))
    #输入，要改成的形状

    #将输入标准化，归一化
    network_input=network_input/float(num_pitch)
    #将期望输出转换成{0，1}布尔矩阵，配合categorical_crossentrogy误差算法的使用
    network_output=tf.keras.utils.to_categorical(network_output)
    #keras中的这个方法可以将一个向量传进去转成布尔矩阵，供交叉熵的计算
    return network_input,network_output

#if __name__ == '__main__':
#   println(34);
#  train()
