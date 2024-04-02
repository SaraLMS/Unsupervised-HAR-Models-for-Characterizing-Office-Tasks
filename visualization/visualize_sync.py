# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from load.load_sync_data import load_data_from_csv
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def visualize_sync_signals(file_path: str) -> None:
    # load signals
    df = load_data_from_csv(file_path)

    # plot axis to check synchronization
    # y-axis from phone acc and -1 * x-axis from watch acc
    plt.figure(figsize=(10, 6))
    plt.plot(df['yAcc'], label='y-axis acc phone')
    plt.plot(-1 * df['xAcc_wear'], label='- x-axis acc watch')
    plt.title('Synchronized signals')
    plt.legend()
    plt.grid(True)
    plt.show()
