import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import _VF
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import PackedSequence


class PeepholeLSTM(nn.LSTM):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        proj_size=0,
        device=None,
        dtype=None,
    ):
        super().__init__(
            input_size,
            hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
            proj_size=0,
            device=None,
            dtype=None,
        )
        self._init_peephole_weights()

    def forward(self, input, hx=None):
        self._update_flat_weights()

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        do_permute = False
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            if hx is None:
                h_zeros = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    real_hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                c_zeros = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                hx = (h_zeros, c_zeros)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"LSTM: Expected input to be 2D or 3D, got {input.dim()}D instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                h_zeros = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    real_hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                c_zeros = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                hx = (h_zeros, c_zeros)
                self.check_forward_args(input, hx, batch_sizes)
            else:
                if is_batched:
                    if hx[0].dim() != 3 or hx[1].dim() != 3:
                        msg = (
                            "For batched 3-D input, hx and cx should "
                            f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = (
                            "For unbatched 2-D input, hx and cx should "
                            f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                self.check_forward_args(input, hx, batch_sizes)
                hx = self.permute_hidden(hx, sorted_indices)

        if batch_sizes is None:
            result = self._peephole_lstm(
                input,
                hx,
                self._flat_weights,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = self._peephole_lstm(
                input,
                batch_sizes,
                hx,
                self._flat_weights,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:  # type: ignore[possibly-undefined]
                output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)

    def _init_peephole_weights(self):
        self.w_ci = nn.Parameter(torch.Tensor(self.hidden_size))
        self.w_cf = nn.Parameter(torch.Tensor(self.hidden_size))
        self.w_co = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.w_ci, -0.1, 0.1)
        nn.init.uniform_(self.w_cf, -0.1, 0.1)
        nn.init.uniform_(self.w_co, -0.1, 0.1)

    def _peephole_lstm(self, input, hx, *args):
        def _lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh, w_ci, w_cf, w_co):
            hx, cx = hidden
            gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate + w_ci * cx)
            forgetgate = torch.sigmoid(forgetgate + w_cf * cx)
            cellgate = torch.tanh(cellgate)
            cy = (forgetgate * cx) + (ingate * cellgate)
            outgate = torch.sigmoid(outgate + w_co * cy)
            hy = outgate * torch.tanh(cy)
            return hy, cy

        if isinstance(input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_seq_length = batch_sizes.size(0)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            max_seq_length = input.size(1) if self.batch_first else input.size(0)

        if hx is None:
            hx = input.new_zeros(self.num_layers, max_batch_size, self.hidden_size)
            cx = input.new_zeros(self.num_layers, max_batch_size, self.hidden_size)
        else:
            hx, cx = hx

        layer_output = None
        new_hx = []
        new_cx = []

        for layer in range(self.num_layers):
            layer_input = input if layer == 0 else layer_output

            w_ih = self._flat_weights[layer * 4]
            w_hh = self._flat_weights[layer * 4 + 1]
            b_ih = self._flat_weights[layer * 4 + 2]
            b_hh = self._flat_weights[layer * 4 + 3]

            layer_hx = hx[layer]
            layer_cx = cx[layer]

            if batch_sizes is None:
                layer_output = []
                for t in range(max_seq_length):
                    layer_hx, layer_cx = _lstm_cell(
                        layer_input[t],
                        (layer_hx, layer_cx),
                        w_ih,
                        w_hh,
                        b_ih,
                        b_hh,
                        self.w_ci[layer],
                        self.w_cf[layer],
                        self.w_co[layer],
                    )
                    layer_output.append(layer_hx)
                layer_output = torch.stack(layer_output)
            else:
                layer_output = []
                for t in range(max_seq_length):
                    batch_size = batch_sizes[t]
                    if batch_size != layer_input.size(0):
                        layer_input = layer_input[:batch_size]
                        layer_hx = layer_hx[:batch_size]
                        layer_cx = layer_cx[:batch_size]
                    layer_hx, layer_cx = _lstm_cell(
                        layer_input[t],
                        (layer_hx, layer_cx),
                        w_ih,
                        w_hh,
                        b_ih,
                        b_hh,
                        self.w_ci[layer],
                        self.w_cf[layer],
                        self.w_co[layer],
                    )
                    layer_output.append(layer_hx)
                layer_output = torch.cat(layer_output)

            new_hx.append(layer_hx)
            new_cx.append(layer_cx)

        layer_output = (
            PackedSequence(layer_output, batch_sizes, sorted_indices, unsorted_indices)
            if isinstance(input, PackedSequence)
            else layer_output
        )
        return layer_output, (torch.stack(new_hx), torch.stack(new_cx))
