import tkinter as tk
from tkinter import filedialog, messagebox, font, Label
from PIL import Image, ImageTk, ImageEnhance
import pickle
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.src.utils import pad_sequences
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import speech_recognition as sr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa.display

with open('word_dict.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('label_dict.pk', 'rb') as f:
    label_dictionary = pickle.load(f)

scaler = joblib.load('scaler_mlp.pkl')
speech_model = load_model('model_cnn.h5')
text_model = load_model('model_word_lstm.h5')
encoder = joblib.load('encoder.joblib')


def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs, y, sr


def plot_waveform(audio, ax, canvas_agg):
    ax.clear()
    ax.plot(audio)
    ax.set_title("Audio Waveform")
    canvas_agg.draw()


def plot_mfcc(audio, sr, ax, canvas_agg):
    ax.clear()
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    print(mfccs.shape)
    S = librosa.power_to_db(mfccs)
    librosa.display.specshow(S, sr=sr, x_axis='time', ax=ax)
    ax.set_title('MFCC')
    canvas_agg.draw()


def audio_to_text(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print("You said: {}".format(text))
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""


def extract_features(audio_file):
    # 提取音频特征
    y, sr = librosa.load(audio_file)
    result = np.array([])

    # 提取你所需要的所有特征
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(y))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    result = np.hstack((result, mel))

    return result


def audio_predic(audio_path, model):
    features = extract_features(audio_path)
    features = scaler.transform([features])  # 标准化特征

    # 更改形状以匹配模型输入
    features = np.expand_dims(features, axis=2)

    # 预测情感
    prediction = speech_model.predict(features)
    predicted_index = np.argmax(prediction)  # 获取预测最高分的索引
    if model == 1:
        increase_prob_labels = ['angry', 'disgust', 'fear', 'sad']
    else:
        increase_prob_labels = ['happy', 'neutral']
    # 定义增加的概率值
    increase_amount = 0.01  # 增加0.01的概率
    # print("All class probabilities:")
    # print(prediction)
    for idx, prob in enumerate(prediction[0]):
        # 创建一个与类别数量相同的全零数组，并将预测索引设为1
        dummy_input = np.zeros((1, len(prediction[0])))
        dummy_input[0, idx] = 1
        class_label = encoder.inverse_transform(dummy_input)[0][0]
        if class_label in increase_prob_labels:
            # 增加概率
            prediction[0][idx] += increase_amount
        print(f"{class_label}: {prob:.4f}")

    predicted_index = np.argmax(prediction)
    # 输出最高预测概率
    print(f"Predicted probability: {prediction[0][predicted_index]:.4f}")

    probilities = "Predicted probability: {class_probability} %".format(
        class_probability=prediction[0][predicted_index] * 100)

    # 将数字标签转换为文本标签
    predicted_label = encoder.inverse_transform(prediction)[0, 0]
    print(predicted_label)

    return predicted_label, probilities


def text_sentiment_classification(text, input_shape=180):
    sequence = [word_dictionary[word] for word in text if word in word_dictionary]

    # 序列填充
    padded_sequence = pad_sequences([sequence], maxlen=input_shape, padding='post', value=0)

    # 模型预测
    prediction = text_model.predict(padded_sequence)

    # 将预测结果转换为标签
    label_index = np.argmax(prediction)
    sentiment_label = [label for label, index in label_dictionary.items() if index == label_index][0]

    return sentiment_label


def classify():
    file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
    if file_path:
        text = audio_to_text(file_path)
        print(text)
        if text != "":
            sentiment = text_sentiment_classification(text)
            print(sentiment)
            if sentiment == "正面":
                ans, probabilities = audio_predic(file_path, 2)
            else:
                ans, probabilities = audio_predic(file_path, 1)

            text_pre.config(text="识别语音文本: {text}".format(text=text))

            print(ans, probabilities)
            result_text.config(
                text="Classification: {ans}\n{class_probability}".format(ans=ans, class_probability=probabilities))

            y, sr = librosa.load(file_path)
            mfccs, y, sr = extract_mfcc(file_path)

            fig, axs = plt.subplots(2)
            canvas_agg = FigureCanvasTkAgg(fig, master=right_frame)

            # 绘制波形图和MFCC图
            plot_waveform(y, axs[0], canvas_agg)
            plot_mfcc(y, sr, axs[1], canvas_agg)
            plt.close(fig)

            for widget in right_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=right_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack()


def img_transparent(image_path, transparency_level=0.5):
    with Image.open(image_path) as img:
        # 分离透明度并调整
        alpha = img.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(transparency_level)
        img.putalpha(alpha)
        return img


root = tk.Tk()
root.geometry("1053x540")
root.title('Speech Emotion Recognition')

photo = tk.PhotoImage(file='logo.gif')
# 设置为窗口的图标（这种方法可能不会改变窗口左上角的小图标）
root.tk.call('wm', 'iconphoto', root._w, photo)
root.iconphoto(False, photo)

bg_image = img_transparent("hello.png", transparency_level=0.5)
photo = ImageTk.PhotoImage(bg_image)

background_label = tk.Label(root, image=photo)
background_label.place(relwidth=1, relheight=1)
# 保留对图片的引用，避免垃圾回收
background_label.image = photo

# Title
title_label = tk.Label(root, text="LPrincess's Emotion System", font=("Comic Sans MS", 15, "bold"))
title_label.pack(side="top")

# Left Frame
left_frame = tk.Frame(root, width=640, height=540)
left_frame.pack(side="left", fill="both", expand=True)
bg_image_l = img_transparent("hello_l.png", transparency_level=0.5)
photo_l = ImageTk.PhotoImage(bg_image_l)
background_label_l = tk.Label(left_frame, image=photo_l)
background_label_l.place(relwidth=1, relheight=1)
background_label_l.image = photo_l

right_frame = tk.Frame(root, width=640, height=540)
right_frame.pack(side="right", fill="both", expand=True)
bg_image_r = img_transparent("hello_r.png", transparency_level=0.5)
photo_r = ImageTk.PhotoImage(bg_image_r)
background_label_r = tk.Label(right_frame, image=photo_r)
background_label_r.place(relwidth=1, relheight=1)
background_label_r.image = photo_r

upload_label = tk.Label(left_frame, text="上传*.wav音频文件：", font=("Helvetica", 12))
upload_label.pack(pady=10)

button_image = Image.open("button.png")
button_photo = ImageTk.PhotoImage(button_image)
upload_button = tk.Button(left_frame, image=button_photo, command=classify, borderwidth=0)
upload_button.image = button_photo
upload_button.pack(pady=10)

text_pre = tk.Label(left_frame, text="识别语音文本：", font=("Helvetica", 12))
text_pre.pack(pady=20)

result_text = tk.Label(left_frame, text="预测结果：", font=("Helvetica", 12))
result_text.pack(pady=20)

canvas = tk.Canvas(right_frame)
canvas.pack(pady=20)

root.resizable(False, False)

root.mainloop()
