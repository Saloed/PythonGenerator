from net_conf import *

IS_TRAIN = True

DECODER_STATE_SIZE = ENCODER_STATE_SIZE * 2

if not IS_TRAIN:
    DROPOUT_PROB = 1.0
    ENCODER_DROPOUT_PROB = 1.0

DATA_SET_BASE_DIR = '../django_data_set/data/'
DATA_SET_NAME = 'django_data_set'

MODEL_SAVE_PATH = 'trained/'
MODEL_BASE_NAME = 'model'

PRETRAIN_SAVE_PATH = 'pretrained/'
PRETRAIN_BASE_NAME = 'pretrain'
