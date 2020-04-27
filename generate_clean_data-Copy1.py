

#########################################################################
# Copyright Eduardo Fonseca 2018, v1.0
# This software is distributed under the terms of the MIT License
#
# If you use this code or part of it, please cite the following paper:
# Eduardo Fonseca, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, Xavier Serra, "Learning Sound Event
# Classifiers from Web Audio with Noisy Labels", in Proc. IEEE ICASSP 2019, Brighton, UK, 2019
#
#########################################################################

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange
import time
import pprint
import datetime
import argparse
from scipy.stats import gmean
import yaml

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from keras import backend as K


import utils
from feat_ext import load_audio_file, get_mel_spectrogram, modify_file_variable_length
from data import get_label_files, DataGeneratorPatch, PatchGeneratorPerFile, DataGeneratorPatchBinary, DataGeneratorFileFeatures
from architectures import get_model_baseline, get_model_binary
from eval import Evaluator
from losses import lq_loss_wrap, crossentropy_max_wrap, crossentropy_outlier_wrap, crossentropy_reed_wrap,\
    crossentropy_max_origin_wrap, crossentropy_outlier_origin_wrap, lq_loss_origin_wrap, crossentropy_reed_origin_wrap


target_label = 4
positive_threshold = 0.9
add_criterion = 9

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

model_path = 'model/generate_clean_data/thres_%.2f_crit_%d/label%d/' % (positive_threshold, add_criterion, target_label)
record_path = 'record/generate_clean_data/thres_%.2f_crit_%d/' % (positive_threshold, add_criterion)

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(record_path):
    os.makedirs(record_path)


start = time.time()

now = datetime.datetime.now()
print("Current date and time:")
print(str(now))

# =========================================================================================================

# ==================================================================== ARGUMENTS
parser = argparse.ArgumentParser(description='Code for ICASSP2019 paper Learning Sound Event Classifiers from Web Audio'
                                             ' with Noisy Labels')
parser.add_argument('-p', '--params_yaml',
                    dest='params_yaml',
                    action='store',
                    required=False,
                    type=str)
args = parser.parse_args()
print('\nYaml file with parameters defining the experiment: %s\n' % str(args.params_yaml))



# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables

# Read parameters file from yaml passed by argument
params = yaml.load(open('config/params.yaml'))
params_ctrl = params['ctrl']
params_extract = params['extract']
params_learn = params['learn']
params_loss = params['loss']
params_recog = params['recognizer']

suffix_in = params['suffix'].get('in')
suffix_out = params['suffix'].get('out')


params_extract['audio_len_samples'] = int(params_extract.get('fs') * params_extract.get('audio_len_s'))
#

# ======================================================== PATHS FOR DATA, FEATURES and GROUND TRUTH
# where to look for the dataset
path_root_data = params_ctrl.get('dataset_path')

params_path = {'path_to_features': os.path.join(path_root_data, 'features'),
               'featuredir_tr': 'audio_train_varup2/',
               'featuredir_te': 'audio_test_varup2/',
               'path_to_dataset': path_root_data,
               'audiodir_tr': 'FSDnoisy18k.audio_train/',
               'audiodir_te': 'FSDnoisy18k.audio_test/',
               'audio_shapedir_tr': 'audio_train_shapes/',
               'audio_shapedir_te': 'audio_test_shapes/',
               'gt_files': os.path.join(path_root_data, 'FSDnoisy18k.meta')}


params_path['featurepath_tr'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_tr'))
params_path['featurepath_te'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_te'))

params_path['audiopath_tr'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_tr'))
params_path['audiopath_te'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_te'))

params_path['audio_shapepath_tr'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_tr'))
params_path['audio_shapepath_te'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_te'))


# ======================================================== SPECIFIC PATHS TO SOME IMPORTANT FILES
# ground truth, load model, save model, predictions, results
params_files = {'gt_test': os.path.join(params_path.get('gt_files'), 'test.csv'),
                'gt_train': os.path.join(params_path.get('gt_files'), 'train.csv')}

# # ============================================= print all params to keep record in output file
print('\nparams_ctrl=')
pprint.pprint(params_ctrl, width=1, indent=4)
print('params_files=')
pprint.pprint(params_files, width=1, indent=4)
print('params_extract=')
pprint.pprint(params_extract, width=1, indent=4)
print('params_learn=')
pprint.pprint(params_learn, width=1, indent=4)
print('params_loss=')
pprint.pprint(params_loss, width=1, indent=4)
print('params_recog=')
pprint.pprint(params_recog, width=1, indent=4)
print('\n')


# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA
# ============================================================== READ TRAIN and TEST DATA

# aim: lists with all wav files for tr and te
train_csv = pd.read_csv(params_files.get('gt_train'))
test_csv = pd.read_csv(params_files.get('gt_test'))
filelist_audio_tr = train_csv.fname.values.tolist()
filelist_audio_te = test_csv.fname.values.tolist()

# get positions of manually_verified clips: separate between CLEAN and NOISY sets
filelist_audio_tr_flagveri = train_csv.manually_verified.values.tolist()
idx_flagveri = [i for i, x in enumerate(filelist_audio_tr_flagveri) if x == 1]
idx_flagnonveri = [i for i, x in enumerate(filelist_audio_tr_flagveri) if x == 0]

# create list of ids that come from the noisy set
noisy_ids = [int(filelist_audio_tr[i].split('.')[0]) for i in idx_flagnonveri]
params_learn['noisy_ids'] = noisy_ids

# get positions of clips of noisy_small subset
# subset of the NOISY set of comparable size to that of CLEAN
filelist_audio_tr_nV_small_dur = train_csv.noisy_small.values.tolist()
idx_nV_small_dur = [i for i, x in enumerate(filelist_audio_tr_nV_small_dur) if x == 1]

# create dict with ground truth mapping with labels:
# -key: path to wav
# -value: the ground truth label too
file_to_label = {params_path.get('audiopath_tr') + k: v for k, v in
                 zip(train_csv.fname.values, train_csv.label.values)}

# ========================================================== CREATE VARS FOR DATASET MANAGEMENT
# list with unique n_classes labels and aso_ids
list_labels = sorted(list(set(train_csv.label.values)))
list_aso_ids = sorted(list(set(train_csv.aso_id.values)))

# create dicts such that key: value is as follows
# label: int
# int: label
label_to_int = {k: v for v, k in enumerate(list_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}

# create ground truth mapping with categorical values
file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}


file_name = 'record/generate_clean_data/train.csv'
train_csv_clean = pd.read_csv(file_name)
binary_labels = np.array(train_csv_clean[str(target_label)])
positive_list = np.where(binary_labels==1)[0]
negative_list = np.where(binary_labels==0)[0]


te_files = [f for f in os.listdir(params_path.get('featurepath_tr')) if f.endswith(suffix_in + '.data') and
          os.path.isfile(os.path.join(params_path.get('featurepath_tr'), f.replace(suffix_in, suffix_out)))]

for iteration in range(5):
    # to store predictions
    te_preds = np.empty((len(te_files), 10))
    list_preds = []
    model_list = []
    for model_j in range(10):
        print('iteration:%d,model:%d'%(iteration,model_j))
        train_idx_neg = np.random.choice(negative_list, len(positive_list), replace=False)
        train_idx = list(positive_list) + list(train_idx_neg)
        ff_list_tr = [filelist_audio_tr[i].replace('.wav', suffix_in + '.data') for i in train_idx]
        labels_audio_train = np.concatenate((np.ones((len(positive_list), 1), dtype=np.float32), np.zeros((len(positive_list), 1), dtype=np.float32)), axis=0)


        # sanity check
        print('Number of clips considered as train set: {0}'.format(len(ff_list_tr)))
        print('Number of labels loaded for train set: {0}'.format(len(labels_audio_train)))

        # split the val set randomly (but stratified) within the train set
        tr_files, val_files = train_test_split(ff_list_tr,
                                               stratify=labels_audio_train,
                                               random_state=42
                                               )

        # to improve data generator
        tr_gen_patch = DataGeneratorPatchBinary(labels=labels_audio_train,
                                                feature_dir=params_path.get('featurepath_tr'),
                                                file_list=ff_list_tr,
                                                params_learn=params_learn,
                                                params_extract=params_extract,
                                                suffix_in='_mel',
                                                suffix_out='_label',
                                                floatx=np.float32
                                                )


        # ============================================================DEFINE AND FIT A MODEL
        # ============================================================DEFINE AND FIT A MODEL
        # ============================================================DEFINE AND FIT A MODEL

        tr_loss, val_loss = [0] * params_learn.get('n_epochs'), [0] * params_learn.get('n_epochs')
        # ============================================================
        if params_ctrl.get('learn'):

            model = get_model_binary(params_learn=params_learn, params_extract=params_extract)
            if iteration > 0:
                modelfile = os.path.join(model_path, 'model%d.h5' % model_j)
                model.load_weights(modelfile)

            opt = Adam(lr=params_learn.get('lr'))
            model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

            # callbacks
            hist = model.fit_generator(tr_gen_patch,
                                       steps_per_epoch=tr_gen_patch.nb_iterations,
                                       epochs=params_learn.get('n_epochs'),
                                       class_weight=None,
                                       workers=4,
                                       verbose=2,
                                       )
            
            modelfile = os.path.join(model_path, 'model%d.h5' % model_j)
            model.save_weights(modelfile)
#             model_list.append(model)
    
    
    # ==================================================================================================== PREDICT
    # ==================================================================================================== PREDICT
    # ==================================================================================================== PREDICT


    te_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_tr'),
                                 file_list=te_files,
                                 params_extract=params_extract,
                                 suffix_in='_mel',
                                 floatx=np.float32,
                                 scaler=tr_gen_patch.scaler
                                 )
    
    for i in trange(len(te_files), miniters=int(len(te_files) / 100), ascii=True, desc="Predicting..."):
        # return all patches for a sound file
        patches_file = te_gen_patch.get_patches_file()
        for model_j in range(10):
            modelfile = os.path.join(model_path, 'model%d.h5' % model_j)
            model.load_weights(modelfile)
            # predicting now on the T_F patch level (not on the wav clip-level)
            preds_patch_list = model.predict(patches_file).tolist()
            preds_patch = np.array(preds_patch_list)
            preds_file = np.mean(preds_patch, axis=0)
       
            te_preds[i, model_j] = preds_file        
                
    K.clear_session()
    tf.reset_default_graph()
                
    pos_pred_valid = np.sum(te_preds >= positive_threshold, axis=1)
    add_index = np.where(pos_pred_valid >= add_criterion)[0]
    
    file_name = record_path + '/iteration%d.csv'%iteration
    if not os.path.exists(file_name):        
        train_csv_clean = pd.read_csv('record/generate_clean_data/train.csv')
    else:
        train_csv_clean = pd.read_csv(file_name)
    binary_labels = np.array(train_csv_clean[str(target_label)])
    binary_labels[add_index] = 1
    train_csv_clean[str(target_label)] = binary_labels
    train_csv_clean.to_csv(file_name, index=False)
    positive_list = np.where(binary_labels==1)[0]
    negative_list = np.where(binary_labels==0)[0]
    
    