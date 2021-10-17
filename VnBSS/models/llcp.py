import torch
import numpy as np
from torch import nn
import torchvision
import torch.nn.functional as F
import torch.autograd.profiler as profiler


class Audio_Model(nn.Module):
    def __init__(self, last_shape=8):
        super(Audio_Model, self).__init__()

        # Audio model layers , name of layers as per table 1 given in paper.

        self.conv1 = nn.Conv2d(
            2,
            96,
            kernel_size=(1, 7),
            padding=self.get_padding((1, 7), (1, 1)),
            dilation=(1, 1),
        )

        self.conv2 = nn.Conv2d(
            96,
            96,
            kernel_size=(7, 1),
            padding=self.get_padding((7, 1), (1, 1)),
            dilation=(1, 1),
        )

        self.conv3 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (1, 1)),
            dilation=(1, 1),
        )

        self.conv4 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (2, 1)),
            dilation=(2, 1),
        )

        self.conv5 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (4, 1)),
            dilation=(4, 1),
        )

        self.conv6 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (8, 1)),
            dilation=(8, 1),
        )

        self.conv7 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (16, 1)),
            dilation=(16, 1),
        )

        self.conv8 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (32, 1)),
            dilation=(32, 1),
        )

        self.conv9 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (1, 1)),
            dilation=(1, 1),
        )

        self.conv10 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (2, 2)),
            dilation=(2, 2),
        )

        self.conv11 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (4, 4)),
            dilation=(4, 4),
        )

        self.conv12 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (8, 8)),
            dilation=(8, 8),
        )

        self.conv13 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (16, 16)),
            dilation=(16, 16),
        )

        self.conv14 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (32, 32)),
            dilation=(32, 32),
        )

        self.conv15 = nn.Conv2d(
            96,
            last_shape,
            kernel_size=(1, 1),
            padding=self.get_padding((1, 1), (1, 1)),
            dilation=(1, 1),
        )

        # Batch normalization layers

        self.batch_norm1 = nn.BatchNorm2d(96)
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.batch_norm4 = nn.BatchNorm2d(96)
        self.batch_norm5 = nn.BatchNorm2d(96)
        self.batch_norm6 = nn.BatchNorm2d(96)
        self.batch_norm7 = nn.BatchNorm2d(96)
        self.batch_norm8 = nn.BatchNorm2d(96)
        self.batch_norm9 = nn.BatchNorm2d(96)
        self.batch_norm10 = nn.BatchNorm2d(96)
        self.batch_norm11 = nn.BatchNorm2d(96)
        self.batch_norm11 = nn.BatchNorm2d(96)
        self.batch_norm12 = nn.BatchNorm2d(96)
        self.batch_norm13 = nn.BatchNorm2d(96)
        self.batch_norm14 = nn.BatchNorm2d(96)
        self.batch_norm15 = nn.BatchNorm2d(last_shape)

    def get_padding(self, kernel_size, dilation):
        padding = (
            ((dilation[0]) * (kernel_size[0] - 1)) // 2,
            ((dilation[1]) * (kernel_size[1] - 1)) // 2,
        )
        return padding

    def forward(self, input_audio):
        # input audio will be (2,256,256)

        output_layer = F.relu(self.batch_norm1(self.conv1(input_audio)))
        output_layer = F.relu(self.batch_norm2(self.conv2(output_layer)))
        output_layer = F.relu(self.batch_norm3(self.conv3(output_layer)))
        output_layer = F.relu(self.batch_norm4(self.conv4(output_layer)))
        output_layer = F.relu(self.batch_norm5(self.conv5(output_layer)))
        output_layer = F.relu(self.batch_norm6(self.conv6(output_layer)))
        output_layer = F.relu(self.batch_norm7(self.conv7(output_layer)))
        output_layer = F.relu(self.batch_norm8(self.conv8(output_layer)))
        output_layer = F.relu(self.batch_norm9(self.conv9(output_layer)))
        output_layer = F.relu(self.batch_norm10(self.conv10(output_layer)))
        output_layer = F.relu(self.batch_norm11(self.conv11(output_layer)))
        output_layer = F.relu(self.batch_norm12(self.conv12(output_layer)))
        output_layer = F.relu(self.batch_norm13(self.conv13(output_layer)))
        output_layer = F.relu(self.batch_norm14(self.conv14(output_layer)))
        output_layer = F.relu(self.batch_norm15(self.conv15(output_layer)))

        # output_layer will be (N,8,256,256)
        # we want it to be (N,8*256,256,1)
        batch_size = output_layer.size(0)  # N
        height = output_layer.size(2)  # 256

        output_layer = output_layer.transpose(-1, -2).reshape((batch_size, -1, height, 1))
        return output_layer


class Video_Model(nn.Module):
    def __init__(self, last_shape=256, upsample=True):
        super(Video_Model, self).__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(
            512,
            256,
            kernel_size=(7, 1),
            padding=self.get_padding((7, 1), (1, 1)),
            dilation=(1, 1),
        )

        self.conv2 = nn.Conv2d(
            256,
            256,
            kernel_size=(5, 1),
            padding=self.get_padding((5, 1), (1, 1)),
            dilation=(1, 1),
        )

        self.conv3 = nn.Conv2d(
            256,
            256,
            kernel_size=(5, 1),
            padding=self.get_padding((5, 1), (2, 1)),
            dilation=(2, 1),
        )

        self.conv4 = nn.Conv2d(
            256,
            256,
            kernel_size=(5, 1),
            padding=self.get_padding((5, 1), (4, 1)),
            dilation=(4, 1),
        )

        self.conv5 = nn.Conv2d(
            256,
            256,
            kernel_size=(5, 1),
            padding=self.get_padding((5, 1), (8, 1)),
            dilation=(8, 1),
        )

        self.conv6 = nn.Conv2d(
            256,
            256,
            kernel_size=(5, 1),
            padding=self.get_padding((5, 1), (16, 1)),
            dilation=(16, 1),
        )

        # Batch normalization layers

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(last_shape)

    def get_padding(self, kernel_size, dilation):
        padding = (
            ((dilation[0]) * (kernel_size[0] - 1)) // 2,
            ((dilation[1]) * (kernel_size[1] - 1)) // 2,
        )
        return padding

    def forward(self, input_video):
        # input video will be (512,100)
        if len(input_video.shape) == 3:
            input_video = input_video.unsqueeze(-1)

        output_layer = F.relu(self.batch_norm1(self.conv1(input_video)))
        output_layer = F.relu(self.batch_norm2(self.conv2(output_layer)))
        output_layer = F.relu(self.batch_norm3(self.conv3(output_layer)))
        output_layer = F.relu(self.batch_norm4(self.conv4(output_layer)))
        output_layer = F.relu(self.batch_norm5(self.conv5(output_layer)))
        output_layer = F.relu(self.batch_norm6(self.conv6(output_layer)))

        # for upsampling , as mentioned in paper
        if self.upsample:
            output_layer = nn.functional.interpolate(output_layer, size=(256, 1), mode="nearest")

        return output_layer


# so now , video_output is (N,256,256,1)
# and audio_output is  (N,8*256,256,1)
# where N = batch_size


class Llcp(nn.Module):
    """Audio Visual Speech Separation model as described in [1].
    All default values are the same as paper.

        Args:
            num_person (int): total number of persons (as i/o).
            device (torch.Device): device used to return the final tensor.
            audio_last_shape (int): relevant last shape for tensor in audio network.
            video_last_shape (int): relevant last shape for tensor in video network.
            input_spectrogram_shape (tuple(int)): shape of input spectrogram.

        References:
            [1]: 'Looking to Listen at the Cocktail Party:
            A Speaker-Independent Audio-Visual Model for Speech Separation' Ephrat et. al
            https://arxiv.org/abs/1804.03619
    """

    def __init__(
            self,
            num_person=2,
            audio_last_shape=8,
            video_last_shape=256,
            input_spectrogram_shape=(256, 256, 2),
    ):
        super(Llcp, self).__init__()
        self.num_person = num_person
        self.input_dim = (
                audio_last_shape * input_spectrogram_shape[1] + video_last_shape
        )

        self.audio_output = Audio_Model(last_shape=audio_last_shape)
        self.video_output = Video_Model(last_shape=video_last_shape)

        self.lstm = nn.LSTM(
            self.input_dim,
            400,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(400, 600)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(600, 600)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(600, 600)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.complex_mask_layer = nn.Linear(600, 2 * 256)
        torch.nn.init.xavier_uniform_(self.complex_mask_layer.weight)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)

        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(256)

    def forward(self, input_audio, input_video):
        # input_audio will be (N,2,256,256)
        # input_video will be of size (N,512,100)

        input_audio = input_audio.transpose(2, 3)  # (N,2,256,256)
        audio_out = self.audio_output(input_audio)
        # audio_out will be (N,8*256,256,1)
        AVFusion = [audio_out]

        video_out = self.video_output(input_video)
        AVFusion.append(video_out)

        mixed_av = torch.cat(AVFusion, dim=1)

        mixed_av = mixed_av.squeeze(3)  # (N,input_dim,256)
        mixed_av = torch.transpose(mixed_av, 1, 2)  # (N,256,input_dim)

        self.lstm.flatten_parameters()
        mixed_av, (h, c) = self.lstm(mixed_av)
        mixed_av = mixed_av[..., :400] + mixed_av[..., 400:]

        mixed_av = self.batch_norm1((F.relu(self.fc1(mixed_av))))
        mixed_av = self.drop1(mixed_av)

        mixed_av = self.batch_norm2(F.relu(self.fc2(mixed_av)))
        mixed_av = self.drop2(mixed_av)

        mixed_av = self.batch_norm3(F.relu(self.fc3(mixed_av)))  # (N,256,600)
        mixed_av = self.drop3(mixed_av)

        complex_mask = self.complex_mask_layer(mixed_av)  # (N,256(T),2(C)*256(F)*num_person)

        batch_size = complex_mask.size(0)  # N
        complex_mask = complex_mask.view(batch_size, 256, 2, 256).transpose(1, 2)


        return complex_mask.transpose(2, 3)  # (B,C,F,T,P)
