# ------------------------------------------------------------------------------------------------------------------- #
# subject information and clustering
# ------------------------------------------------------------------------------------------------------------------- #

SUBJECT_PREFIX = "P0"
CLASS = "class"
SUBJECT = "subject"
SUBCLASS = "subclass"
SUBJECT_ID = "subject_id"
FEATURE_SET = "feature_set"
# ------------------------------------------------------------------------------------------------------------------- #
# supported file extensions
# ------------------------------------------------------------------------------------------------------------------- #

TXT = ".txt"
CSV = ".csv"

# ------------------------------------------------------------------------------------------------------------------- #
# synchronization constants
# ------------------------------------------------------------------------------------------------------------------- #
BUFFER_SIZE_SECONDS = 0.5

# synchronization methods
CROSSCORR = "crosscorr"
TIMESTAMPS = "timestamps"

# ------------------------------------------------------------------------------------------------------------------- #
# supported sensors and devices
# ------------------------------------------------------------------------------------------------------------------- #
# supported devices
PHONE = "phone"
WATCH = "watch"
MBAN = "mban"
SUPPORTED_DEVICES = [PHONE, WATCH, MBAN]

WEAR_ACCELEROMETER = "WEAR_ACCELEROMETER"
ACCELEROMETER = "ACCELEROMETER"

# supported sensors for each device
ACC = "acc"
GYR = "gyr"
MAG = "mag"
ROTVEC = "rotvec"
NOISE = "noise"
WEARHEARTRATE = "wearheartrate"
EMG = "emg"
SEC = 'sec'

SUPPORTED_PHONE_SENSORS = [ACC, GYR, MAG, ROTVEC, NOISE]
SUPPORTED_WATCH_SENSORS = [ACC, GYR, MAG, ROTVEC, WEARHEARTRATE]
# mban for reference only - not implemented
SUPPORTED_MBAN_SENSORS = [ACC, EMG]

# column dataframes prefixes
ACCELEROMETER_PREFIX = "Acc"
GYROSCOPE_PREFIX = "Gyr"
MAGNETOMETER_PREFIX = "Mag"
ROTATION_VECTOR_PREFIX = "Rot"
WEAR_PREFIX = "_wear"
SUPPORTED_PREFIXES = [ACCELEROMETER_PREFIX, GYROSCOPE_PREFIX, MAGNETOMETER_PREFIX, ROTATION_VECTOR_PREFIX]

# ------------------------------------------------------------------------------------------------------------------- #
# supported activities
# ------------------------------------------------------------------------------------------------------------------- #

# activity types
WALKING = "walking"
STANDING = "standing"
SITTING = "sitting"
CABINETS = "cabinets"
STAIRS = "stairs"

SUPPORTED_ACTIVITIES = [WALKING, STANDING, SITTING, CABINETS, STAIRS]

# ------------------------------------------------------------------------------------------------------------------- #
# supported segment suffixes
# ------------------------------------------------------------------------------------------------------------------- #

WALKING_SUFFIXES = ['_slow', '_medium', '_fast']

STAIRS_4SUFFIXES = ['_stairsup1', '_stairsdown1', '_stairsup2', '_stairsdown2']

STAIRS_8SUFFIXES = ['_stairsup1', '_stairsdown1', '_stairsup2', '_stairsdown2',
                    '_stairsup3', '_stairsdown3', '_stairsup4', '_stairsdown4']

STANDING_SUFFIXES = ['_stand_still1', '_gestures', '_stand_still2']

CABINETS_SUFFIXES = ['_coffee', '_folders']

SITTING_SUFFIXES = ['_sit']

# ------------------------------------------------------------------------------------------------------------------- #
# supported clustering models
# ------------------------------------------------------------------------------------------------------------------- #

KMEANS = "kmeans"
AGGLOMERATIVE = "agglomerative"
GAUSSIAN_MIXTURE_MODEL = "gmm"
SUPPORTED_MODELS = [KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL]
