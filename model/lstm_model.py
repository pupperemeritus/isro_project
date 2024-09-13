from typing import Tuple


from pytorch_lightning import LightningModule
import torch
import torch.nn as nn


class LSTMForecastModel(LightningModule):

    def __init__(self, input_dim, hidden_dim, grid_shape, output_size):
        super(LSTMForecastModel, self).__init__()
        self.grid_shape = grid_shape
        self.flattened_grid_size = grid_shape[0] * grid_shape[1]

        self.lstm_s4 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_s4 = nn.Linear(hidden_dim, self.flattened_grid_size * output_size)

        self.lstm_phase = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_phase = nn.Linear(hidden_dim, self.flattened_grid_size * output_size)

    def forward(self, x_s4, x_phase):
        batch_size, seq_len, _ = x_s4.shape

        lstm_out_s4, _ = self.lstm_s4(x_s4)
        output_s4 = self.fc_s4(lstm_out_s4[:, -1, :])  # Use only the last output
        output_s4 = output_s4.view(
            batch_size, -1, self.grid_shape[0], self.grid_shape[1]
        )

        lstm_out_phase, _ = self.lstm_phase(x_phase)
        output_phase = self.fc_phase(
            lstm_out_phase[:, -1, :]
        )  # Use only the last output
        output_phase = output_phase.view(
            batch_size, -1, self.grid_shape[0], self.grid_shape[1]
        )

        return output_s4, output_phase

    def training_step(self, batch, batch_idx):
        features_s4, features_phase = batch["features_s4"], batch["features_phase"]
        target_s4, target_phase = batch["target_s4"], batch["target_phase"]

        predictions_s4, predictions_phase = self(features_s4, features_phase)

        loss_s4 = torch.nn.functional.mse_loss(predictions_s4, target_s4)
        loss_phase = torch.nn.functional.mse_loss(predictions_phase, target_phase)

        self.log("train_loss_s4", loss_s4)
        self.log("train_loss_phase", loss_phase)
        return {
            "train_loss_s4": loss_s4,
            "train_loss_phase": loss_phase,
            "loss": loss_s4 + loss_phase,
        }

    def validation_step(self, batch, batch_idx):
        features_s4, features_phase = batch["features_s4"], batch["features_phase"]
        target_s4, target_phase = batch["target_s4"], batch["target_phase"]

        predictions_s4, predictions_phase = self(features_s4, features_phase)

        loss_s4 = torch.nn.functional.mse_loss(predictions_s4, target_s4)
        loss_phase = torch.nn.functional.mse_loss(predictions_phase, target_phase)

        self.log("val_loss_s4", loss_s4)
        self.log("val_loss_phase", loss_phase)
        return {
            "val_loss_s4": loss_s4,
            "val_loss_phase": loss_phase,
            "loss": loss_s4 + loss_phase,
        }

    def test_step(self, batch, batch_idx):
        features_s4, features_phase = batch["features_s4"], batch["features_phase"]
        target_s4, target_phase = batch["target_s4"], batch["target_phase"]

        predictions_s4, predictions_phase = self(features_s4, features_phase)

        loss_s4 = torch.nn.functional.mse_loss(predictions_s4, target_s4)
        loss_phase = torch.nn.functional.mse_loss(predictions_phase, target_phase)

        self.log("test_loss_s4", loss_s4)
        self.log("test_loss_phase", loss_phase)
        return {
            "test_loss_s4": loss_s4,
            "test_loss_phase": loss_phase,
            "loss": loss_s4 + loss_phase,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
