
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

SUPPORTED_PHONE_SENSORS = [ACC, GYR, MAG, ROTVEC, NOISE]
SUPPORTED_WATCH_SENSORS = [ACC, GYR, MAG, ROTVEC, WEARHEARTRATE]
# mban for reference only - not implemented
SUPPORTED_MBAN_SENSORS = [ACC, EMG]

# column dataframes prefixes
ACCELEROMETER_PREFIX = "Acc"
GYROSCOPE_PREFIX = "Gyr"
