from net_conf import *

IS_TRAIN = False

RULES_DECODER_STATE_SIZE = RULES_QUERY_ENCODER_STATE_SIZE * 2
WORDS_DECODER_STATE_SIZE = RULES_ENCODER_STATE_SIZE * 2

if not IS_TRAIN:
    DROPOUT_PROB = 1.0
    ENCODER_DROPOUT_PROB = 1.0

DJANGO_DATA_SET_TYPE = 'DJANGO'
HS_DATA_SET_TYPE = 'HS'

DATA_SET_TYPE = DJANGO_DATA_SET_TYPE

DATA_SET_BASE_DIR = '../django_data_set/data/'
DATA_SET_NAME = 'django_data_set_v2'
FULL_DATA_SET_NAME = 'django.cleaned.dataset.freq3.par_info.refact.space_only.order_by_ulink_len.bin'

MODEL_SAVE_PATH = 'trained/'
RULES_MODEL_BASE_NAME = 'model_rules'
WORDS_MODEL_BASE_NAME = 'model_words'

BEST_RULES_MODEL_BASE_NAME = RULES_MODEL_BASE_NAME + '-4'
BEST_WORDS_MODEL_BASE_NAME = WORDS_MODEL_BASE_NAME + '-3'

PRETRAIN_SAVE_PATH = 'pretrained/'
PRETRAIN_BASE_NAME = 'pretrain'

RULES_BEAM_SIZE = 30
