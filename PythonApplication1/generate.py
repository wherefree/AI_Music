
import pickle
import numpy as np
import tensorflow as tf 
from network import *
from utils import *
"""
用训练好的神经网络模型参数来作曲 
"""
#以之前所得的最佳参数来生成音乐
def generate():
    gpus=tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
    #加载用于训练神经网络的音乐数据
    with open('data/notes','rb') as filepath:#以读的方式打开文件
        notes=pickle.load(filepath)
    #得到所有不重复的音符的名字和数目
    pitch_names=sorted(set(item for item in notes))
    num_pitch=len(set(notes))
    print(num_pitch)
    print('\n')
    network_input,normalized_input=prepare_sequences(notes,pitch_names,num_pitch)

    #载入之前训练是最好的参数（最小loss），来生成神经网络模型
    model=network_model(normalized_input,num_pitch,"weights-71-0.0908.hdf5")

    #用神经网络来生成音乐数据
    prediction=generate_notes(model,network_input,pitch_names,num_pitch)

    #用预测的音乐数据生成midi文件
    create_music(prediction)


def prepare_sequences(notes,pitch_names,num_pitch):
    #从midi中读取的notes和所有音符的数量
    """ 
    为神经网络提供好要训练的序列 
    """
    sequence_length=100#序列长度

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
        network_output.append([pitch_to_int[sequence_out]])
        #把sequence_out的一个字符转为整数
    
    n_patterns=len(network_input)#输入序列长度
    
    #将输入序列的形状转成神经网络模型可以接受的
    normalized_input=np.reshape(network_input,(n_patterns,sequence_length,1))
    #输入，要改成的形状

    #将输入标准化，归一化
    normalized_input=normalized_input/float(num_pitch)
    return (network_input,normalized_input)

def generate_notes(model,network_input,pitch_names,num_pitch):
    """
    基于序列音符，用神经网络来生成新的音符 
    """
    #从输入里随机选择一个序列，作为“预测”/生成的音乐的起始点
    start=np.random.randint(0,len(network_input)-1)#从0到神经网络输入-1中随机选择一个整数

    #创建一个字典用于映射 整数 和 音调，和训练相反的操作
    int_to_pitch=dict((num,pitch) for num,pitch in enumerate(pitch_names))

    pattern=network_input[start]#随机选择的序列起点

    #神经网络实际生成的音符
    prediction_output=[]

    #生成700个音符
    for note_index in range(700):
        prediction_input=np.reshape(pattern,(1,len(pattern),1))
        #输入，归一化
        prediction_input=prediction_input/float(num_pitch)

        #读取参数文件，载入训练所得最佳参数文件的神经网络来预测新的音符
        prediction=model.predict(prediction_input,verbose=0)#根据输入预测结果
        
        #argmax取最大的那个维度（类似One-hot编码）
        index=np.argmax(prediction)
        result=int_to_pitch[index]
        prediction_output.append(result)

        #start往后移动
        pattern.append(index)
        pattern=pattern[1:len(pattern)]
    return prediction_output
    
if __name__ == '__main__':
    generate()