import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class LSTMForecastModel(LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        prediction_horizon,
        grid_lon_steps,  # Added
        grid_lat_steps,  # Added
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * prediction_horizon)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Debugging statements
        print(f"Input shape: {x.shape}")
        print(f"Expected input size: {self.lstm.input_size}")

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]

        # Use the last output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Fully connected layer
        out = self.fc(last_output)  # [batch_size, output_size * prediction_horizon]

        # Reshape output
        batch_size = x.shape[0]
        prediction_horizon = self.hparams.prediction_horizon
        grid_lon_steps = self.hparams.grid_lon_steps
        grid_lat_steps = self.hparams.grid_lat_steps

        out = out.view(
            batch_size,
            prediction_horizon,
            grid_lon_steps,
            grid_lat_steps,
            2,  # S4 and Phase
        )

        return out

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch, batch_idx, stage):
        x_s4, x_phase = batch["features_s4"], batch["features_phase"]
        y_s4, y_phase = batch["target_s4"], batch["target_phase"]

        # Combine S4 and phase data
        x = torch.cat(
            (x_s4, x_phase), dim=-1
        )  # [batch, seq_len, lat_steps, lon_steps * 2]

        # Flatten spatial dimensions
        batch_size, seq_len, lat_steps, lon_steps_times_2 = x.shape
        x = x.view(
            batch_size, seq_len, lat_steps * lon_steps_times_2
        )  # [batch, seq_len, input_size]

        # Forward pass
        y_hat = self(x)  # [batch, pred_horizon, grid_lon, grid_lat, 2]

        # Separate predictions for S4 and phase
        y_hat_s4 = y_hat[..., 0]  # [batch, pred_horizon, grid_lon, grid_lat]
        y_hat_phase = y_hat[..., 1]  # [batch, pred_horizon, grid_lon, grid_lat]

        # Calculate loss
        loss_s4 = self.criterion(y_hat_s4, y_s4)
        loss_phase = self.criterion(y_hat_phase, y_phase)
        total_loss = loss_s4 + loss_phase

        # Log metrics
        self.log(
            f"{stage}_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_loss_s4", loss_s4, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_loss_phase",
            loss_phase,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
