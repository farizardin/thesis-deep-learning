from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import sys

from cv2 import cv2
import mediapipe as mp
import json
import glob
from pathlib import Path
import pprint as pp
from keypoints_normalization.image_cropping import ImageCroppingNormalization
from keypoints_normalization.recalculate_keypoints2 import RecalculateNormalization2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
num_joint = 115
max_frame = 300
num_person_out = 1
num_person_in = 1


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Shift Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--keypoints_normalization_method',
        default='coordinate_recalculation',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    parser.add_argument('--video', default=None)
    parser.add_argument('--data_path', default=None)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        arg.model_saved_name = "./save_models/"+arg.Experiment_name
        arg.work_dir = "./work_dir/"+arg.Experiment_name
        self.arg = arg
        # self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input(
                            'Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        # self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_epoch = 0

    # def load_data(self):
    #     Feeder = import_class(self.arg.feeder)
    #     self.data_loader = dict()
    #     self.print_log("NUM WORKER: "+str(self.arg.num_worker))
    #     if self.arg.phase == 'train':
    #         self.data_loader['train'] = torch.utils.data.DataLoader(
    #             dataset=Feeder(**self.arg.train_feeder_args),
    #             batch_size=self.arg.batch_size,
    #             shuffle=True,
    #             num_workers=self.arg.num_worker,
    #             drop_last=True,
    #             worker_init_fn=init_seed)
    #     self.data_loader['test'] = torch.utils.data.DataLoader(
    #         dataset=Feeder(**self.arg.test_feeder_args),
    #         batch_size=self.arg.test_batch_size,
    #         shuffle=False,
    #         num_workers=self.arg.num_worker,
    #         drop_last=False,
    #         worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            # self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            # for w in self.arg.ignore_weights:
            #     if weights.pop(w, None) is not None:
            #         self.print_log('Sucessfully Remove Weights: {}.'.format(w))
            #     else:
            #         self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4
                if 'Linear_weight' in key:
                    weight_decay = 1e-3
                elif 'Mask' in key:
                    weight_decay = 0.0

                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult,
                            'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def eval(self, data, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # print("eval")
        self.model.eval()
        # self.print_log('Eval epoch: {}'.format(epoch + 1))

        loss_value = []
        score_frag = []
        right_num_total = 0
        total_num = 0
        loss_total = 0
        # step = 0
        # process = tqdm(self.data_loader[ln])
        # for batch_idx, (data, label, index) in enumerate(process):
        data = torch.from_numpy(data)
        data = Variable(
            data.float().cuda(self.output_device),
            requires_grad=False,
            volatile=True)
        # label = Variable(
        #     label.long().cuda(self.output_device),
        #     requires_grad=False,
        #     volatile=True)

        with torch.no_grad():
            output = self.model(data)

        # loss = self.loss(output, label)
        score_frag.append(output.data.cpu().numpy())
        # loss_value.append(loss.data.cpu().numpy())

        _, predict_label = torch.max(output.data, 1)
        # step += 1

        # print("data inside ", data.data.cpu().numpy().shape)

        # if wrong_file is not None or result_file is not None:
        #     predict = list(predict_label.cpu().numpy())
        #     true = list(label.data.cpu().numpy())
        #     for i, x in enumerate(predict):
        #         if result_file is not None:
        #             f_r.write(str(x) + ',' + str(true[i]) + '\n')
        #         if x != true[i] and wrong_file is not None:
        #             f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
        score = np.concatenate(score_frag)
        return predict_label, score


class Feeder_hsd(Dataset):
    """ Feeder for skeleton-based hand sign recognition in HandSign dataset
    # 21 keypoints list:
    # https://google.github.io/mediapipe/images/mobile/hand_landmarks.png

    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=1,
                 num_person_out=1):
        self.data = data
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        self.sample_name = ["test_webcam"]
        # print(self.sample_name)

        # output data shape (N, C, T, V, M)
        self.N = 1  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        video_info = self.data

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                # score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::3]
                data_numpy[1, frame_index, :, m] = pose[1::3]
                data_numpy[2, frame_index, :, m] = pose[2::3]

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # sort by score
        # sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        # for t, s in enumerate(sort_index):
        #     data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
        #                                                                0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy


def gendata(data,
            num_person_in=num_person_in,  # observe the first 1 persons
            num_person_out=num_person_out,  # then choose 1 persons with the highest score
            max_frame=max_frame):
    feeder = Feeder_hsd(
        data=data,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = np.zeros((1, 3, max_frame, num_joint,
                  num_person_out), dtype=np.float32)

    for i, _ in enumerate(sample_name):
        data = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data

    # np.save(data_out_path, fp)
    return fp


def make_json(data, cls, is_kinetics):
    ord_A = ord("A")
    dict_data = {"data": [], "label": cls, "label_index": ord(cls)-ord_A}
    z_data = []
    for i, frame in enumerate(data):
        dict_data["data"].append(
            {"frame_index": i+1, "skeleton": [{"pose": []}]})
        for keypoint in frame:
            dict_data["data"][i]["skeleton"][0]["pose"].append(
                round(keypoint[0], 4))
            dict_data["data"][i]["skeleton"][0]["pose"].append(
                round(keypoint[1], 4))
            z = round(abs(keypoint[2]), 4)
            if not is_kinetics:
                dict_data["data"][i]["skeleton"][0]["pose"].append(z)
            else:
                z_data.append(z)

        if is_kinetics:
            dict_data["data"][i]["skeleton"][0]["score"] = z_data

    return dict_data


def get_rank(data, index):
    ord_A = ord("A")
    return [[chr(ord_A+index[0]), data[index[0]]], [chr(ord_A+index[1]), data[index[1]]], [chr(ord_A+index[2]), data[index[2]]], [chr(ord_A+index[3]), data[index[3]]], [chr(ord_A+index[4]), data[index[4]]]]


def levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    return min(levenshtein(a[1:], b[1:])+(a[0] != b[0]),
               levenshtein(a[1:], b)+1,
               levenshtein(a, b[1:])+1)


labels = ['akan',
        'anda',
        'apa',
        'atau',
        'baca',
        'bagaimana',
        'bahwa',
        'beberapa',
        'besar',
        'bisa',
        'buah',
        'dan',
        'dari',
        'dengan',
        'dia',
        'haus',
        'ingin',
        'ini',
        'itu',
        'jadi',
        'juga',
        'kami',
        'kata',
        'kecil',
        'kumpul',
        'labuh',
        'lain',
        'laku',
        'lapar',
        'main',
        'makan',
        'masing',
        'mereka',
        'milik',
        'minum',
        'oleh',
        'pada',
        'rumah',
        'satu',
        'saya',
        'sebagai',
        'tambah',
        'tangan',
        'tetapi',
        'tidak',
        'tiga',
        'udara',
        'untuk',
        'waktu',
        'yang']

parser = get_parser()

# load arg form config file
p = parser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        default_arg = yaml.load(f)
    key = vars(p).keys()
    pp.pprint(key)
    pp.pprint(default_arg)
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    parser.set_defaults(**default_arg)

arg = parser.parse_args()
processor = Processor(arg)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


x_start = 200
y_start = 200

width = 320
height = 256

x_end = x_start + width
y_end = y_start + height


fps = 30
frame_counter = 0
ord_A = ord("A")
dict_data = {"data":[],"label": "?","label_index":ord("?")-ord_A}
xyz = []
final_pred = ""


# Capture video from webcam
if p.video is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(p.video)

frame_process = 120
frame_counter = 0
chr_pred = "process.."

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
recalculation = RecalculateNormalization2(mp_holistic)
image_crop = ImageCroppingNormalization(holistic)
predicted_array = []
with holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        frame_counter += 1
        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if arg.keypoints_normalization_method == "image_cropping":
                coordinates_data = image_crop.normalize(image)
            elif arg.keypoints_normalization_method == "coordinate_recalculation":
                    coordinates_data = recalculation.normalize(image)
            dict_data["data"].append({"frame_index": frame_counter, "skeleton":[{"pose":[]}]})
            dict_data["data"][frame_counter - 1]["skeleton"][0]["pose"] = coordinates_data
            key = cv2.waitKey(1)
            if frame_counter == frame_process:
                ord_A = ord("A")
                np_data = gendata(dict_data,max_frame=max_frame,num_person_in=num_person_in,num_person_out=num_person_out)
                predicted, score = processor.eval(np_data)
                chr_pred = chr(ord_A + predicted)
                print("PREDICTED:", labels[predicted])
                print("SCORE:",score)
                len_final_pred = len(final_pred)

                if len_final_pred == 0:
                    final_pred +=chr_pred
                elif final_pred[len_final_pred-1] != chr_pred:
                    final_pred +=chr_pred
                
                if len(predicted_array) != 0:
                    if predicted != predicted_array[-1]:
                        predicted_array.append(predicted)
                else:
                    predicted_array.append(predicted)


                frame_counter = 0
                xyz=[]
                dict_data = {"data":[],"label": "?","label_index":ord("?")-ord_A}
            if key == ord("q"):
                break
        else:
            break
cap.release()
cv2.destroyAllWindows()

pred_str = ""
tokenized_pred = []
pred_len = len(predicted_array)
if pred_len > 0:
    j = 0
    for i in predicted_array:
        pred_str += labels[i]
        tokenized_pred.append(labels[i])
        if j != pred_len:
            pred_str += " "
        j += 1

if p.video:
    print("FINAL PREDICTION:", pred_str)
    vid_name = p.video.split("/")[-1].split(".")[0]
    vid_name = vid_name.split("_")
    print(vid_name, tokenized_pred)
    print("ACTUAL:", vid_name)
    print("Levenstein Distance:", levenshtein(tokenized_pred, vid_name))
