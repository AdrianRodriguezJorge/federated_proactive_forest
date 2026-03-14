"""Adaptador para NSL-KDD (KDDTrain+.csv / KDDTest+.csv)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class",
]
CAT_COLS  = ["protocol_type", "service", "flag"]
FEAT_COLS = [c for c in COLUMNS if c != "class"]


class NslKddAdapter(IDatasetAdapter):
    def __init__(self, train_path: str, test_path: str,
                 scale: bool = True, scaler_type: str = "standard"):
        self.train_path = train_path
        self.test_path  = test_path
        self.scale      = scale
        self.scaler_type = scaler_type
        self._encoder   = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
        self._label_encoder = LabelEncoder()
        if scale:
            self._scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        else:
            self._scaler = None
        self._class_names_: list = []

    @property
    def name(self) -> str:
        return "nslkdd"

    @property
    def n_classes(self) -> int:
        return len(self._class_names_)

    def _read(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, skiprows=1, header=None, names=COLUMNS[:42])
        # Algunos CSVs de NSL-KDD traen columna extra de dificultad
        if df.shape[1] == 43:
            df = df.iloc[:, :42]
        return df

    def load(self) -> DatasetSplit:
        train_df = self._read(self.train_path)
        test_df  = self._read(self.test_path)

        train_df[CAT_COLS] = self._encoder.fit_transform(train_df[CAT_COLS])
        test_df[CAT_COLS]  = self._encoder.transform(test_df[CAT_COLS])

        X_train = train_df[FEAT_COLS].values.astype(np.float64)
        X_test  = test_df[FEAT_COLS].values.astype(np.float64)
        y_train_raw = train_df["class"].values
        y_test_raw  = test_df["class"].values

        # Mantener y como strings
        y_train = y_train_raw
        y_test  = y_test_raw

        if self._scaler:
            X_train = self._scaler.fit_transform(X_train)
            X_test  = self._scaler.transform(X_test)

        self._class_names_ = sorted(list(set(y_train_raw)))  # clases únicas en orden

        return DatasetSplit(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=FEAT_COLS,
            class_names=self._class_names_,
            dataset_name=self.name,
        )
