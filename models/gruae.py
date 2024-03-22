import torch
import torch.nn as nn

from .image_encoder.cnn import EncoderCNN


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        return x, hidden


class DecoderCNN(nn.Module):
    def __init__(self, latent_dim):
        super(DecoderCNN, self).__init__()
        self.decoder_embed = nn.Linear(latent_dim, 512*4*4)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder_embed(x)
        x = x.view(-1, 512, 4, 4)
        x = self.network(x)
        return x


class GRUAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(GRUAutoEncoder, self).__init__()
        self.latent_dim = cfg.MODEL.LATENT_DIM
        self.infer = cfg.MODEL.EVAL
        self.img_size = cfg.INPUT.IMAGE_SIZE

        self.image_encoder = EncoderCNN(self.latent_dim)
        self.encoder_gru = GRU(self.latent_dim, self.latent_dim, num_layers=1)
        self.decoder_gru = GRU(self.latent_dim, self.latent_dim, num_layers=1)
        self.decoder_cnn = DecoderCNN(self.latent_dim)

    def forward(self, x_encoder, x_decoder=None):
        self.batch_size = x_encoder.size()[0]
        self.seq_length = x_encoder.size()[1]

        x_encoder = x_encoder.view(-1, 3, self.img_size[0], self.img_size[1])
        x_encoder = self.image_encoder(x_encoder)
        x_encoder = x_encoder.reshape(self.batch_size, self.seq_length, -1)

        h = torch.randn(1, self.batch_size, self.latent_dim, requires_grad=True)
        h = h.to(x_encoder.device)

        encoder_output, encoder_hidden = self.encoder_gru(x_encoder, h)
        if self.infer:
            del x_encoder, x_decoder
            return encoder_output, encoder_hidden

        x_decoder = x_decoder.view(-1, 3, self.img_size[0], self.img_size[1])
        x_decoder = self.image_encoder(x_decoder)
        x_decoder = x_decoder.reshape(self.batch_size, self.seq_length, -1)

        decoder_output, decoder_hidden = self.decoder_gru(x_decoder, encoder_hidden)
        rec = self.decoder_cnn(decoder_output)
        rec = rec.view(self.batch_size, self.seq_length, 3, self.img_size[0], self.img_size[1])

        del x_encoder, x_decoder
        del encoder_output, encoder_hidden
        del decoder_output, decoder_hidden
        return rec
