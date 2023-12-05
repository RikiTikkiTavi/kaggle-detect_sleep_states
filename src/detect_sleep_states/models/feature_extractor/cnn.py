from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ref: https://github.com/analokmaus/kaggle-g2net-public/tree/main/models1d_pytorch
class CNNSpectrogram(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            base_filters: int | tuple = 128,
            kernel_sizes: tuple = (32, 16, 4, 2),
            stride: int = 4,
            sigmoid: bool = False,
            output_size: Optional[int] = None,
            conv: Callable = nn.Conv1d,
            reinit: bool = True,
            skip_connections_features: Sequence[int] = tuple(),
            required_height=32,
            features_to_drop_idx: Sequence[int] = (2, 3),
            features_drop_p: float = 0.3
    ):
        """
        :param in_channels: Number of classes
        :param base_filters: Each output channel will have this height (H) (where width is L).
        :param kernel_sizes: Kernel sizes
        :param stride: Stride
        :param sigmoid:
        :param output_size: L timesteps (L) in output. Use to downsample
        :param conv:
        :param reinit:
        """
        super().__init__()
        # Out channels equals to kernel sizes
        self.out_chans = len(kernel_sizes) + 1 if len(skip_connections_features) > 0 else len(kernel_sizes)
        self.auto_out_chans = len(kernel_sizes)
        self.out_size = output_size
        self.sigmoid = sigmoid
        self.skip_connections_features = skip_connections_features
        self.required_height = required_height
        self.features_to_drop_idx = features_to_drop_idx
        # Set base filters in correct format
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])
        self.height = required_height
        # Init list of modules
        self.spec_conv = nn.ModuleList()
        # Iterate over channels
        self.drop_feature = nn.Dropout1d(p=features_drop_p, inplace=False)
        for i in range(self.auto_out_chans):
            # Single convolutional layer with kernel size corresponding to channel
            # Produces H features from combinations of input features
            tmp_block = [
                conv(
                    in_channels,
                    base_filters[0],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=(kernel_sizes[i] - 1) // 2,
                )
            ]
            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j],
                            base_filters[j + 1],
                            kernel_size=kernel_sizes[i],
                            stride=stride,
                            padding=(kernel_sizes[i] - 1) // 2,
                        ),
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                self.spec_conv.append(tmp_block[0])

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        if len(self.skip_connections_features) > 0:
            self.short_path = nn.Sequential(
                nn.AdaptiveMaxPool1d(output_size=self.out_size)
            )

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (_type_): (batch_size, in_channels, time_steps)

        Returns:
            _type_: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        x[:, self.features_to_drop_idx, :] = self.drop_feature(x[:, self.features_to_drop_idx, :])
        out: list[torch.Tensor] = []
        for i in range(self.auto_out_chans):
            # (batch_size, 1, height, time_steps)
            out_channel = self.spec_conv[i](x)
            out.append(out_channel)

        img = torch.stack(out, dim=1)  # (batch_size, out_chans, height, time_steps)

        if self.out_size is not None:
            img = self.pool(img)  # (batch_size, out_chans, height, out_size)
        if self.sigmoid:
            img = img.sigmoid()

        if len(self.skip_connections_features) > 0:
            x_skip = self.short_path(x[:, self.skip_connections_features, :])
            if x_skip.size(1) < img.size(2):
                height_diff = img.size(2) - x_skip.size(1)
                pad_top = height_diff // 2
                pad_bot = height_diff - pad_top
                x_skip = F.pad(x_skip, (0, 0, pad_top, pad_bot), value=-1)
            x_skip = torch.unsqueeze(x_skip, 1)
            img = torch.cat((img, x_skip), dim=1)

        # Pad to required_height
        if img.size(2) < self.required_height:
            height_diff = self.required_height - img.size(2)
            pad_top = height_diff // 2
            pad_bot = height_diff - pad_top
            img = F.pad(img, (0, 0, pad_top, pad_bot), value=-1)

        return img
