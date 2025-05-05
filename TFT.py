import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import copy
from pathlib import Path
import warnings
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)


path_train = "../df_train.parquet"
path_test = "../df_test.parquet"

df_train = pd.read_parquet(path_train)
df_test = pd.read_parquet(path_test)

df_train["carb"] = df_train["carb"].fillna(0)
df_train["bolus"] = df_train["bolus"].fillna(0)
df_train["basal_rate"] = df_train["basal_rate"].ffill().bfill()

df_train["time_idx"] = (df_train["Zulu Time"] - df_train["Zulu Time"].min()).dt.total_seconds().astype(int)
df_train["time_idx"] = df_train.groupby("group_id").cumcount()

df_test["carb"] = df_test["carb"].fillna(0)
df_test["bolus"] = df_test["bolus"].fillna(0)
df_test["basal_rate"] = df_test["basal_rate"].ffill().bfill()
df_test["time_idx"] = df_test.groupby("group_id").cumcount()

from pytorch_forecasting.models import TemporalFusionTransformer

max_encoder_length = 6
max_prediction_length = 1

training = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target="Value",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx", "carb", "bolus", "basal_rate"],
    time_varying_unknown_reals=["Value"]
)

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df_train, test_size=0.2, shuffle=False)

train = TimeSeriesDataSet.from_dataset(training, df_train, predict=False)
val = TimeSeriesDataSet.from_dataset(training, df_val, predict=False)

train_dataloader = train.to_dataloader(train=True, batch_size=128, num_workers=0)
val_dataloader = val.to_dataloader(train=False, batch_size=128, num_workers=0)

from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=8,
    attention_head_size=1,
    dropout=0.1,
    loss=QuantileLoss(),
    log_interval=-1,
    reduce_on_plateau_patience=4,
)

from pytorch_lightning import Trainer


trainer = pl.Trainer(max_epochs=30, accelerator="gpu")
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

