import pickle
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Embedding,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(filepath,input_shape=20):
    df=pd.read_csv(filepath)

    # 标签及词汇表
    labels,vocabulary=list(df['label'].unique()),list(df['evaluation'].unique())

    # 构造字符级别的特征
    string=''
    for word in vocabulary:
        string+=word

    vocabulary=set(string)

    # 字典列表
    word_dictionary={word:i+1 for i,word in enumerate(vocabulary)}
    with open('word_dict.pk','wb') as f:
        pickle.dump(word_dictionary,f)
    inverse_word_dictionary={i+1:word for i,word in enumerate(vocabulary)}
    label_dictionary={label:i for i,label in enumerate(labels)}
    with open('label_dict.pk','wb') as f:
        pickle.dump(label_dictionary,f)
    output_dictionary={i:labels for i,labels in enumerate(labels)}

    # 词汇表大小
    vocab_size=len(word_dictionary.keys())
    # 标签类别数量
    label_size=len(label_dictionary.keys())

    # 序列填充，按input_shape填充，长度不足的按0补充
    x=[[word_dictionary[word] for word in sent] for sent in df['evaluation']]
    x=pad_sequences(maxlen=input_shape,sequences=x,padding='post',value=0)
    y=[[label_dictionary[sent]] for sent in df['label']]
    y=[np_utils.to_categorical(label,num_classes=label_size) for label in y]
    y=np.array([list(_[0]) for _ in y])

    return x,y,output_dictionary,vocab_size,label_size,inverse_word_dictionary

# 创建深度学习模型，Embedding + LSTM + Softmax
def create_LSTM(n_units,input_shape,output_dim,filepath):
    x,y,output_dictionary,vocab_size,label_size,inverse_word_dictionary=load_data(filepath)
    model=Sequential()
    model.add(Embedding(input_dim=vocab_size+1,output_dim=output_dim,
                        input_length=input_shape,mask_zero=True))
    model.add(LSTM(n_units,input_shape=(x.shape[0],x.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(label_size,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    plot_model(model,to_file='./model_lstm.png',show_shapes=True)
    # 输出模型信息
    model.summary()

    return model

# 模型训练
def model_train(input_shape,filepath,model_save_path):
    # 将数据集分为训练集和测试集，占比为9：1
    # input_shape=100
    x,y,output_dictionary,vocab_size,label_size,inverse_word_dictionary=load_data(filepath,input_shape)
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.1,random_state=42)

    # 模型输入参数，需要根据自己需要调整
    n_units=100
    batch_size=32
    epochs=20
    output_dim=20

    # 模型训练
    lstm_model=create_LSTM(n_units,input_shape,output_dim,filepath)
    lstm_model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size,verbose=1)

    # 模型保存
    lstm_model.save(model_save_path)

    # 测试条数
    N= test_x.shape[0]
    predict=[]
    label=[]
    for start,end in zip(range(0,N,1),range(1,N+1,1)):
        print(f'start:{start}, end:{end}')
        sentence=[inverse_word_dictionary[i] for i in test_x[start] if i!=0]
        y_predict=lstm_model.predict(test_x[start:end])
        print('y_predict:',y_predict)
        label_predict=output_dictionary[np.argmax(y_predict[0])]
        label_true=output_dictionary[np.argmax(test_y[start:end])]
        print(f'label_predict:{label_predict}, label_true:{label_true}')
        # 输出预测结果
        print(''.join(sentence),label_true,label_predict)
        predict.append(label_predict)
        label.append(label_true)

    history = lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1,
                             validation_data=(test_x, test_y))

    # 预测准确率
    acc=accuracy_score(predict,label)
    print('模型在测试集上的准确率:%s'%acc)
    return history

if __name__=='__main__':
    filepath='data_single.csv'
    input_shape=180
    model_save_path='model_word_lstm.h5'
    model_train(input_shape,filepath,model_save_path)

    history = model_train(input_shape, filepath, model_save_path)
    # 绘图
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()