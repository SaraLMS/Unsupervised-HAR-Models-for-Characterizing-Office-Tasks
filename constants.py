# ------------------------------------------------------------------------------------------------------------------- #
# private functions # TODO ADD THESE SECTIONS
# ------------------------------------------------------------------------------------------------------------------- #
BUFFER_SIZE_SECONDS = 0.5

CROSSCORR = "crosscorr"
TIMESTAMPS = "timestamps"

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

# activity types
WALKING = "walking"
STANDING = "standing"
SITTING = "sitting"
CABINETS = "cabinets"
STAIRS = "stairs"

SUPPORTED_ACTIVITIES = [WALKING, STANDING, SITTING, CABINETS, STAIRS]


# clustering
KMEANS = "kmeans"
AGGLOMERATIVE = "agglomerative"
GAUSSIAN_MIXTURE_MODEL = "gmm"
DBSCAN = "dbscan"
BIRCH = "birch"
SUPPORTED_MODELS = [KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, DBSCAN, BIRCH]
