

import sys
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import types
import torch
import torch.nn as nn
import pyaudio
import wave
import numpy as np
DIR = os.getcwd() + '\\src\\util\\AI'
print(DIR)
sys.path.append(DIR + '\\utils')

labels = ['bat_den_1', 'bat_den_2', 'bat_den_3', 'bat_quat_1', 'bat_quat_2', 'bat_quat_3', 'dong_cua',
          'mo_cua', 'tat_den_1', 'tat_den_2', 'tat_den_3', 'tat_quat_1', 'tat_quat_2', 'tat_quat_3']
import lb

dict_labels = {
    'bat_den_1': [1, 1, 0, 0],
    'bat_den_2': [1, 2, 0, 0],
    'bat_den_3': [1, 3, 0, 0],
    'bat_quat_1': [1, 0, 1, 0],
    'bat_quat_2': [1, 0, 2, 0],
    'bat_quat_3': [1, 0, 3, 0],
    'tat_den_1': [0, 1, 0, 0],
    'tat_den_2': [0, 2, 0, 0],
    'tat_den_3': [0, 3, 0, 0],
    'tat_quat_1': [0, 0, 1, 0],
    'tat_quat_2': [0, 0, 2, 0],
    'tat_quat_3': [0, 0, 3, 0],
    'mo_cua': [1, 0, 0, 1],
    'dong_cua': [0, 0, 0, 1],
}


def load_weights(model, weights, PRINT=False):
    # Load weights into model.
    # If param's name is different, raise error.
    # If param's size is different, skip this param.
    # see: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2

    for i, (name, param) in enumerate(weights.items()):
        model_state = model.state_dict()

        if name not in model_state:
            print("-"*80)
            print("weights name:", name)
            print("RNN states names:", model_state.keys())
            assert 0, "Wrong weights file"

        model_shape = model_state[name].shape
        if model_shape != param.shape:
            print(
                f"\nWarning: Size of {name} layer is different between model and weights. Not copy parameters.")
            print(
                f"\tModel shape = {model_shape}, weights' shape = {param.shape}.")
        else:
            model_state[name].copy_(param)


def set_default_args():

    args = types.SimpleNamespace()

    # model params
    args.input_size = 12  # == n_mfcc
    args.batch_size = 1
    args.hidden_size = 64
    args.num_layers = 3

    # training params
    args.num_epochs = 100
    args.learning_rate = 0.0001
    args.learning_rate_decay_interval = 5  # decay for every 5 epochs
    args.learning_rate_decay_rate = 0.5  # lr = lr * rate
    args.weight_decay = 0.00
    args.gradient_accumulations = 16  # number of gradient accums before step

    # training params2
    args.load_weights_from = None
    args.finetune_model = False  # If true, fix all parameters except the fc layer
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    args.data_folder = "Data_Test/DataTraining/"
    args.train_eval_test_ratio = [0.0, 0.0, 1.0]
    args.do_data_augment = False

    # labels
    # args.classes_txt = "config/classes.names"
    args.classes_txt = labels
    # should be added with a value somewhere, like this:
    args.num_classes = None
    #                = len(lib.read_list(args.classes_txt))

    # log setting
    args.plot_accu = True  # if true, plot accuracy for every epoch
    # if false, not calling plt.show(), so drawing figure in background
    args.show_plotted_accu = False
    args.save_model_to = 'checkpoints/'  # Save model and log file
    # e.g: model_001.ckpt, log.txt, log.jpg

    return args


def create_RNN_model(args, load_weights_from=None):
    ''' A wrapper for creating a 'class RNN' instance '''
    # Update some dependent args
    # args.num_classes = len(lib.read_list(args.classes_txt)) # read from "config/classes.names"
    args.num_classes = len(labels)  # read from "config/classes.names"
    args.save_log_to = args.save_model_to + "log.txt"
    args.save_fig_to = args.save_model_to + "fig.jpg"

    # Create model
    device = args.device
    model = RNN(args.input_size, args.hidden_size, args.num_layers,
                args.num_classes, device).to(device)

    # Load weights
    if load_weights_from:
        print(f"Load weights from: {load_weights_from}")
        weights = torch.load(load_weights_from)
        load_weights(model, weights)

    return model

# Recurrent neural network (many-to-one)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, classes=None):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.classes = classes

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # shape = (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        '''Predict one label from one sample's features'''
        # x: feature from a sample, LxN
        #   L is length of sequency
        #   N is feature dimension
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
        x = x.to(self.device)
        outputs = self.forward(x)
        _, predicted = torch.max(outputs.data, 1)
        predicted_index = predicted.item()
        return predicted_index

    def set_classes(self, classes):
        self.classes = classes

    def predict_audio_label(self, audio):
        idx = self.predict_audio_label_index(audio)
        assert self.classes, "Classes names are not set. Don't know what audio label is"
        label = self.classes[idx]
        return label

    def predict_audio_label_index(self, audio):
        audio.compute_mfcc()
        x = audio.mfcc.T  # (time_len, feature_dimension)
        idx = self.predict(x)
        return idx


def setup_classifier(load_weights_from):
    model_args = set_default_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_RNN_model(model_args, load_weights_from)
    return model


def setup_classes_labels(load_classes_from, model):
    classes = lb.read_list(load_classes_from)
    print(f"{len(classes)} classes: {classes}")
    model.set_classes(classes)


FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

model = setup_classifier(
    load_weights_from=DIR+"/checkpoints//025.ckpt")
setup_classes_labels(
    load_classes_from=DIR+"/config/classes.names", model=model)


def prediction(file_name):
    import librosa
    import soundfile as sf
    # Resample audio received to 16000 Hz
    audio, sr = librosa.load(
        os.getcwd() + f'\\src\\resources\\audio\\{file_name}.wav', sr=None)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write(os.getcwd() + f'\\src\\resources\\audio_resampled\\{file_name}_resampled.wav',
             audio_resampled, 16000)

    # Redict audio already resampled to 16000 Hz
    audio = lb.AudioClass(
        filename=os.getcwd() + f'\\src\\resources\\audio_resampled\\{file_name}_resampled.wav')

    class_predicted = model.predict_audio_label(audio)
    return class_predicted, dict_labels[class_predicted]
