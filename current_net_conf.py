from net_conf import *

IS_TRAIN = False

DECODER_STATE_SIZE = ENCODER_STATE_SIZE * 2

if not IS_TRAIN:
    DROPOUT_PROB = 1.0
    ENCODER_DROPOUT_PROB = 1.0

DJANGO_DATA_SET_TYPE = 'DJANGO'
HS_DATA_SET_TYPE = 'HS'

DATA_SET_TYPE = DJANGO_DATA_SET_TYPE

DATA_SET_BASE_DIR = '../django_data_set/data/'
DATA_SET_NAME = 'django_data_set'
FULL_DATA_SET_NAME = 'django.cleaned.dataset.freq3.par_info.refact.space_only.order_by_ulink_len.bin'

MODEL_SAVE_PATH = 'trained/'
MODEL_BASE_NAME = 'model-2'

PRETRAIN_SAVE_PATH = 'pretrained/'
PRETRAIN_BASE_NAME = 'pretrain'

DATA_SET_FIELD_NAMES_MAPPING = {
    'id': 'ids',
    'query': 'queries',
    'rules': 'rules_sequences',
    'words': 'words_sequences',
    'words_mask': 'words_sequences_masks',
    'copy': 'copy_sequences',
    'copy_mask': 'copy_sequences_masks',
    'gen_or_copy': 'generate_or_copy_decision_sequences',
}

RULES_BEAM_SIZE = 10
