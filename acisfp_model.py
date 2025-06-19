from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.losses import Huber
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from cxotime import CxoTime
from matplotlib import gridspec
import matplotlib.patheffects as path_effects
from cheta import fetch_sci as fetch


feature_cols = ["1crat", "1cbat"]
feature_cols += [f"orbitephem0_{ax}" for ax in "xyz"]
feature_cols += [f"aoattqt{i}" for i in range(1, 5)]

target_col = "fptemp_11"
fields = [target_col] + feature_cols


def create_sequences(data, sequence_length=100):
    """
    Create sequences of data for LSTM input.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length, :-1])  # all features
        y.append(data[i + sequence_length, -1])  # target at t + sequence_length
    return np.array(X), np.array(y)


class FPTempDataProcessor:
    def __init__(
        self, scaler_x=None, scaler_y=None, sequence_length=100, model_filename=None
    ):
        from sklearn.preprocessing import MinMaxScaler

        if scaler_x is not None:
            self.scaler_x = scaler_x
        else:
            self.scaler_x = MinMaxScaler()
        if scaler_y is not None:
            self.scaler_y = scaler_y
        else:
            self.scaler_y = MinMaxScaler()
        self.sequence_length = sequence_length
        self.model_filename = model_filename
        if model_filename is not None:
            self.model = load_model(model_filename)
        else:
            self.model = None

    @classmethod
    def from_file(cls, filename):
        import joblib

        scaler_x, scaler_y, sequence_length, model_filename = joblib.load(filename)
        return cls(scaler_x, scaler_y, sequence_length, model_filename)

    def fetch_data(self, start, stop):
        """
        Fetch data from the Cheta telemetry database for the specified time range.

        Parameters
        ----------
        start : str, CxoTime, or float
            Start time for data retrieval. Can be a string in 'YYYY:DOY:HH:MM:SS' format,
            a CxoTime object, or a float representing seconds since the beginning
            of the mission.
        stop : str, CxoTime, or float
            Stop time for data retrieval. Can be a string in 'YYYY:DOY:HH:MM:SS' format,
            a CxoTime object, or a float representing seconds since the beginning
            of the mission.
        """
        return fetch.MSIDset(fields, start, stop, stat="5min")

    def process_data(self, msids, fit=False):
        fptemp = np.interp(
            msids["1crat"].times, msids["fptemp_11"].times, msids["fptemp_11"].vals
        )

        df = pd.DataFrame(
            {
                "fptemp_11": fptemp,
            }
        )
        for field in feature_cols:
            df[field] = msids[field].vals

        if fit:
            scaled_features = self.scaler_x.fit_transform(df[feature_cols])
            scaled_target = self.scaler_y.fit_transform(
                df[[target_col]]
            )  # target must be 2D
        else:
            scaled_features = self.scaler_x.transform(df[feature_cols])
            scaled_target = self.scaler_y.transform(df[[target_col]])

        data = np.hstack([scaled_features, scaled_target])

        return msids["1crat"].times, data

    def split_data(self, data, times):
        """
        Split data into training, validation, and test sets.

        Parameters
        ----------
        data: numpy.ndarray
            The data to be split, where the last column is the target variable.
            This should be the output of `process_data`.
        times: numpy.ndarray
            The times corresponding to the data, measured in seconds since the
            start of the mission. This should be the output of `process_data`.
        """
        X, y = create_sequences(data, sequence_length=self.sequence_length)
        split_index = int(len(X) * 0.7)
        X_train, y_train = X[:split_index], y[:split_index]
        X_val, y_val = X[split_index:], y[split_index:]
        split_index2 = int(len(X_val) * 0.5)
        X_val, y_val, X_test, y_test = (
            X_val[:split_index2],
            y_val[:split_index2],
            X_val[split_index2:],
            y_val[split_index2:],
        )
        times_train = times[:split_index]
        times_val = times[split_index:]
        times_val, times_test = times_val[:split_index2], times_val[split_index2:]
        data_splits = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "times_train": times_train,
            "times_val": times_val,
            "times_test": times_test,
        }
        return data_splits

    def make_model(self):
        input_shape = (self.sequence_length, len(feature_cols))
        model = Sequential()
        model.add(
            LSTM(
                128,
                return_sequences=True,
                input_shape=input_shape,
            )
        )
        model.add(Dropout(0.2))
        model.add(
            LSTM(
                64,
                return_sequences=True,
            )
        )
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss=Huber(delta=1.0))

        self.model = model

    def train_model(self, ds):
        history = self.model.fit(
            ds["X_train"],
            ds["y_train"],
            validation_data=(ds["X_val"], ds["y_val"]),
            epochs=20,
            batch_size=32,
        )

        return history

    def inverse_transform(self, y_in):
        return self.scaler_y.inverse_transform(y_in.reshape(-1, 1))[:, 0]

    def test_plots(self, msids):
        times, data = self.process_data(msids)
        X, y = create_sequences(data)
        times1 = CxoTime(times[self.sequence_length :])
        y_scaled = self.model.predict(X)
        y_pred = self.inverse_transform(y_scaled)
        y_true = self.inverse_transform(y)
        plt.rc("font", size=14)
        plt.rc("axes", linewidth=1.5)
        plt.rc("xtick.major", size=5, width=1.5)
        plt.rc("ytick.major", size=5, width=1.5)
        plt.rc("xtick.minor", size=3, width=1.5)
        plt.rc("ytick.minor", size=3, width=1.5)
        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(
            2, 2, width_ratios=[2.5, 1.5], height_ratios=[1, 1], wspace=0.25, hspace=0.4
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax2)
        ax1.plot(times1.datetime, y_true, label="Data", linewidth=1.5)
        ax1.plot(times1.datetime, y_pred, label="Model", linewidth=1.5)
        ax2.plot(y_true - y_pred, y_true, ".")
        ax3.plot(times1.datetime, y_true - y_pred)
        ax3.axhline(0, color="black", linestyle="--", linewidth=1.5)
        ax1.legend()
        x_bins = np.arange(-1.5, 1.6, 0.1)
        ax4.hist(y_true - y_pred, bins=x_bins)
        quants = np.percentile(y_true - y_pred, [1, 34, 50, 68, 99])
        ax4.axvline(quants[0], color="k", linestyle="--", linewidth=1.5)
        ax4.axvline(quants[2], color="k", linestyle="--", linewidth=1.5)
        ax4.axvline(quants[4], color="k", linestyle="--", linewidth=1.5)
        ymin, ymax = ax4.get_ylim()
        ytext = 0.07 * (ymin + ymax)
        for i, level in zip([0, 2, 4], [1, 50, 99]):
            ax4.text(
                quants[i] + 0.07,
                ytext,
                f"{level}% Quantile ({quants[i]:.2f} $^\\circ$C)",
                color="black",
                fontsize=12,
                rotation=90,
                path_effects=[
                    path_effects.Stroke(linewidth=3, foreground="white"),
                    path_effects.Normal(),
                ],
            )
        ax1.set_ylabel(r"FP Temperature ($^\circ$C)")
        ax2.set_ylabel(r"FP Temperature ($^\circ$C)")
        ax3.set_ylabel(r"Residuals ($^\circ$C)")
        ax2.set_xlabel(r"Residuals ($^\circ$C)")
        ax4.set_xlabel(r"Residuals ($^\circ$C)")
        ax4.set_ylabel("N")
        for ax in [ax1, ax3]:
            ax.set_xlabel("Date")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y:%j"))
            ax.tick_params(axis="x", rotation=30)
        return fig

    def save(self, filename):
        model_filename = filename.rsplit(".", 1)[0] + ".keras"
        joblib.dump(
            (
                self.scaler_x,
                self.scaler_y,
                self.sequence_length,
                model_filename,
            ),
            filename,
        )
        self.model.save(model_filename)
