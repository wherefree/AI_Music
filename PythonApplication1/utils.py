import os
import subprocess
import pickle
import glob
from music21 import converter,instrument,note,chord,stream#converter负责转换,乐器，音符，和弦类

def get_notes():
    """ 
    从music_midi目录中的所有MIDI文件里读取note，chord
    Note样例：B4，chord样例[C3,E4,G5],多个note的集合，统称“note”
    """
    notes=[]
    for midi_file in glob.glob("music_midi/*.mid"):
        #读取music_midi文件夹中所有的mid文件,file表示每一个文件
        stream=converter.parse(midi_file)#midi文件的读取，解析，输出stream的流类型
    
        #获取所有的乐器部分，开始测试的都是单轨的
        parts=instrument.partitionByInstrument(stream)
        if parts:#如果有乐器部分，取第一个乐器部分
            notes_to_parse=parts.parts[0].recurse()#递归
        else:
            notes_to_parse=stream.flat.notes#纯音符组成
        for element in notes_to_parse:#notes本身不是字符串类型
            #如果是note类型，取它的音高(pitch)
            if isinstance(element,note.Note):
            #格式例如：E6
                notes.append(str(element.pitch))
            elif isinstance(element,chord.Chord):
            #转换后格式：45.21.78(midi_number)
                notes.append('.'.join(str(n) for n in element.normalOrder))#用.来分隔，把n按整数排序
    # 如果 data 目录不存在，创建此目录
    if not os.path.exists("data"):
        os.mkdir("data")
    #将数据写入data/notes
    with open('data/notes','wb') as filepath:#从路径中打开文件，写入
        pickle.dump(notes,filepath)#把notes写入到文件中
    return notes#返回提取出来的notes列表

def create_music(prediction):#生成音乐函数，训练不用
    """ 用神经网络预测的音乐数据来生成mid文件 """
    offset=0#偏移，防止数据覆盖
    output_notes=[]
    #生成Note或chord对象
    #seven_number(prediction)
    for data in prediction:
        #如果是chord格式：45.21.78
        if ('.' in data) or data.isdigit():#data中有.或者有数字
            note_in_chord=data.split('.')#用.分隔和弦中的每个音
            notes=[]#notes列表接收单音
            for current_note in note_in_chord:
                new_note=note.Note(int(current_note))#把当前音符化成整数，在对应midi_number转换成note
                new_note.storedInstrument=instrument.Piano()#乐器用钢琴
                notes.append(new_note)
            new_chord=chord.Chord(notes)#再把notes中的音化成新的和弦
            new_chord.offset=offset#初试定的偏移给和弦的偏移
            output_notes.append(new_chord)#把转化好的和弦传到output_notes中
            offset+=0.5
        #是note格式：
        else:
            new_note=note.Note(data)#note直接可以把data变成新的note
            new_note.offset=offset
            new_note.storedInstrument=instrument.Piano()#乐器用钢琴
            output_notes.append(new_note)#把new_note传到output_notes中
            offset+=0.5
        #每次迭代都将偏移增加，防止交叠覆盖
        #offset+=0.5
    
    #创建音乐流(stream)
    midi_stream=stream.Stream(output_notes)#把上面的循环输出结果传到流
    print("music already done\n")
    #写入midi文件
    midi_stream.write('midi',fp='output.mid')#最终输出的文件名是output.mid，格式是mid

def seven_number(prediction):
    vis_note={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0}#记录12个音在谱中出现的次数
    q1={}
    for i in range(0,11):#生成note对象 0-12个音
        q1[i]=note.Note(i)
    for data in prediction:#遍历data
        if ('.'in data) or data.indigit():#遇到和旋的情况，即有'.'或数字出现
           note_in_chord=data.split('.')
           for j in note_in_chord:#遍历和弦中的音符
                for k in range(0,11):#检测是否符合12个音符中的某一个
                    if q1[k]==note.Note(int(j)):#符合即++
                        #print(q1[k])
                        vis_note[k]+=1
        else:#单个note
            for k in range(0,11):#检测为12个音中的哪一个
                if q1[k]==note.Note(data):#符合即++
                    #print(q1[k])
                    vis_note[k]+=1
       #排序vis数组，选取前7个作为主音，其他音升调或降调为主
    vis_note1=sorted(vis_note.items(),key=lambda d:d[1],reverse=True)
    print(vis_note1[0])
    for data in prediction:
        if('.'in data) or data.indigit():
            mpte_in_chord=data.split('.')
            for j in note_in_chord:
                data=check_and_change_note(vis_note1,j)
        else :
            data=check_and_change_note(vis_note1,j)
    return prediction

def check_and_change_note(vis,data):
    a=note.Note()