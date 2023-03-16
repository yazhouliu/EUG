from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_CPU = 4     
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.RNG_SEED = 42

_C.MODEL = CN()
_C.MODEL.OUT_STRIDE = 16                    # deeplab output stride
_C.MODEL.SYNC_BN = None                     # whether to use sync bn (for multi-gpu), None == Auto detect
_C.MODEL.FREEZE_BN = False                 


_C.LOSS = CN()
_C.LOSS.TYPE = "SegLoss"           # available losses from utils.loss.py
_C.LOSS.IGNORE_LABEL = 255
_C.LOSS.SIZE_AVG = True
_C.LOSS.BATCH_AVG = True 

_C.EXPERIMENT= CN()
_C.EXPERIMENT.NAME = 'psp+ocr'                   # None == Auto name from date and time 
_C.EXPERIMENT.OUT_DIR = "/home/yazhou/wxy4/EUG/" 
_C.EXPERIMENT.CONFIG_FILE1='/home/yazhou/wxy3/configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py'
_C.EXPERIMENT.CONFIG_FILE2='/home/yazhou/wxy3/configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py'
_C.EXPERIMENT.CHECKPOINT_FILE1='/home/yazhou/wxy3/checkpoints/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth'
_C.EXPERIMENT.CHECKPOINT_FILE2='/home/yazhou/wxy3/checkpoints/ocrnet_hr48_512x1024_40k_cityscapes_20200601_033336-55b32491.pth'
_C.EXPERIMENT.EPOCHS = 100                  # number of training epochs
_C.EXPERIMENT.START_EPOCH = 0
_C.EXPERIMENT.USE_BALANCED_WEIGHTS = False
_C.EXPERIMENT.RESUME_CHECKPOINT = None      # path to resume file (stored checkpoint)
_C.EXPERIMENT.EVAL_INTERVAL = 1             # eval every X epoch
_C.EXPERIMENT.EVAL_METRIC = "AnomalyEvaluator" # available evaluation metrics from utils.metrics.py file
_C.EXPERIMENT.MODEL_NAME = 'eug_base'
_C.INPUT = CN()
_C.INPUT.BASE_SIZE = 896 
_C.INPUT.CROP_SIZE = 896 
_C.INPUT.NORM_MEAN = [0.485, 0.456, 0.406]  # mean for the input image to the net (image -> (0, 1) -> mean/std) 
_C.INPUT.NORM_STD = [0.229, 0.224, 0.225]   # std for the input image to the net (image -> (0, 1) -> mean/std) 
_C.INPUT.BATCH_SIZE_TRAIN = 4          # None = Auto set based on training dataset
_C.INPUT.BATCH_SIZE_TEST = 4             # None = Auto set based on training batch size

_C.AUG = CN()
_C.AUG.RANDOM_CROP_PROB = 0.5               # prob that random polygon (anomaly) will be cut from image vs. random noise
_C.AUG.SCALE_MIN = 0.5
_C.AUG.SCALE_MAX = 2.0
_C.AUG.COLOR_AUG = 0.25

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 0.001
_C.OPTIMIZER.LR_SCHEDULER = "poly"          # choices: ['poly', 'step', 'cos']
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 5e-4
_C.OPTIMIZER.NESTEROV = False

_C.DATASET = CN()
_C.DATASET.TRAIN = "cityscapes_2class"      # choices: ['cityscapes'],
_C.DATASET.VAL = "LaF"                      # choices: ['cityscapes'],
_C.DATASET.TEST = "LaF"                     # choices: ['LaF'],
_C.DATASET.FT = False                       # flag if we are finetuning 



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()

