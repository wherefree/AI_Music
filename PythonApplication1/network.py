#神经网络模型
#RNN-LSTM循环神经网络
import tensorflow as tf 
#构建神经网络模型
def network_model(inputs,num_pitch,weights_file=None):#输入，音符的数量，训练后的参数文件
    #测试时要指定weights_file
    #建立模子
    if weights_file != None :
        print("YES\n")
    model=tf.keras.Sequential()
    #第一层
    model.add(tf.keras.layers.LSTM(
        512,#LSTM层神经元的数目是512，也是LSTM层输出的维度
        input_shape=(inputs.shape[1],inputs.shape[2]),#输入的形状，对于第一个LSTM必须设置
        return_sequences=True#返回控制类型，此时是返回所有的输出序列
        #True表示返回所有的输出序列
        #False表示返回输出序列的最后一个输出
        #在堆叠的LSTM层时必须设置，最后一层LSTM不用设置，默认值为False
    ))
    #第二层和第三层
    model.add(tf.keras.layers.Dropout(0.3))#丢弃30%神经元，防止过拟合
    model.add(tf.keras.layers.LSTM(512,return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))#丢弃30%神经元，防止过拟合
    model.add(tf.keras.layers.LSTM(512))#千万不要丢括号！！！！
    #全连接层
    model.add(tf.keras.layers.Dense(256))#256个神经元的全连接层
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_pitch))#输出的数目等于所有不重复的音调数
    #激活层
    model.add(tf.keras.layers.Activation('softmax'))#Softmax激活函数求概率

    #配置神经网络模型
    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0004))
    #选择的损失函数是交叉熵，用来计算误差。使用对于RNN来说比较优秀的优化器-RMSProp
    #优化器如果使用字符串的话会用默认参数导致效果不好

    if weights_file is not None:
        model.load_weights(weights_file)#就把这些参数加载到模型中，weight_file本身是HDF5文件
    return model  
