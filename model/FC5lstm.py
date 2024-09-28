import torch
from torch import nn
import torchvision
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureFusion(nn.Module):

    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=15, stride=2,
                padding=7, groups=128, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                512, 512, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                1024, 512, kernel_size=11, stride=1,
                padding=5, bias=False),
            nn.BatchNorm2d(512),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                2048, 512, kernel_size=7, stride=1,
                padding=3, groups=128, bias=False),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(512),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                1024, 1024, kernel_size=11, stride=1,
                padding=5, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),  # not shown in paper
        )

        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(512, 512 // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(512 // 16, 512, 1, bias=False)

    def forward(self, x1, x2 ,x3):
        x1_1 = self.branch1(x1)
        x2_1 = self.branch2(x2)
        x3_1 = self.branch3(x3)

        pixavg = torch.mean(x1_1, dim=1, keepdim=True)
        detail = self.sigmoid1(pixavg) * x2_1

        chaavg = self.fc2(self.relu1(self.fc1(self.avg_pool(x3_1))))
        seman =  self.sigmoid2(chaavg) * x2_1
        out = self.conv(torch.cat([detail,seman],dim=1))
        return out

class Encoder(nn.Module):
    """
    CNN_Encoder.
    """
    def __init__(self, NetType='resnet50', encoded_image_size=14, attention_method="ByPixel"):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.attention_method = attention_method

        self.FF = FeatureFusion()

        # resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        net = torchvision.models.inception_v3(pretrained=True, transform_input=False) if NetType == 'inception_v3' else \
              torchvision.models.vgg16(pretrained=True) if NetType == 'vgg16' else \
              torchvision.models.resnet50(pretrained=True) if NetType == 'resnet50' else torchvision.models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        # Specifically, Remove: AdaptiveAvgPool2d(output_size=(1, 1)), Linear(in_features=2048, out_features=1000, bias=True)]

        # modules = list(net.children())[:-2]
        modules = list(net.children())[:-1] if NetType == 'inception_v3' or NetType == 'vgg16' else list(net.children())[:-2]
        # modules = list(net.children())[:-1] if NetType == 'inception_v3' else list(net.children())[:-2]  # -2 for resnet & vgg
        if NetType == 'inception_v3': del modules[13]

        self.net = nn.Sequential(*modules)

        # every block of resnet for fusion
        if NetType == 'resnet50' or NetType == 'resnet101' or NetType == 'resnet152':
            resnet_block1 = list(net.children())[:5]
            self.resnet_block1 = nn.Sequential(*resnet_block1)
            resnet_block2 = list(net.children())[5]
            self.resnet_block2 = nn.Sequential(*resnet_block2)
            resnet_block3 = list(net.children())[6]
            self.resnet_block3 = nn.Sequential(*resnet_block3)
            resnet_block4 = list(net.children())[7]
            self.resnet_block4 = nn.Sequential(*resnet_block4)
            self.conv4 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1)
            self.conv3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
            self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)

        # if self.attention_method == "ByChannel":
        #     self.cnn1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     self.relu = nn.ReLU(inplace=True)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # self.adaptive_pool4 = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # self.adaptive_pool3 = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        # with fusion for resnet
        out1 = self.resnet_block1(images)  # 256
        out2 = self.resnet_block2(out1)  # 512
        out3 = self.resnet_block3(out2)  # 1024
        out4 = self.resnet_block4(out3)  # 2048

        # # FIXME:concat432
        out = self.FF(out2,out3,out4)


        # without fusion
        # out = self.net(images)  # (batch_size, 2048, image_size/32, image_size/32)
        # if self.attention_method == "ByChannel":
        #     out = self.relu(self.bn1(self.cnn1(out)))
        out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14] #FIXME:for fusion
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.net.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.net.children())[5:]:  # FIXME:maybe try 6:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        #attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2))  # (batch_size, pixels, encoder_dim)
        return attention_weighted_encoding, alpha

class CrossAttention(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=8):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossAttention, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, input1, input2):
        # dif_as_kv
        input1 = input1.permute(1, 0, 2)
        input2 = input2.permute(1, 0, 2)
        output_1 = self.cross1(input1, input2)  # (Q,K,V)
        output_1 = output_1.permute(1, 0, 2)
        return output_1
    def cross1(self, input,input2):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input, input2, input2)  # (Q,K,V)
        output = input + self.dropout1(attn_output)
        output = self.activation(self.norm1(output))
        return output


class TVAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, embed_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(TVAttention, self).__init__()
        self.nn1 = nn.Linear(encoder_dim, encoder_dim)  # linear layer to transform encoded image
        self.nn2 = nn.Linear(1000, attention_dim)  # linear layer to transform encoded image
        self.crossatt = CrossAttention(dropout=0.5)


    def forward(self, TextFeature, wordFeature, VisionFeature):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        b, n, channels = TextFeature.size(0), TextFeature.size(1), TextFeature.size(1)


        visions = torch.chunk(VisionFeature,chunks=2,dim=2)
        vision1 = visions[0]
        vision2 = visions[1]
        # vision1 TextFeature

        TextFeature = self.nn2(TextFeature.unsqueeze(1))
        # sim_mapv_W = torch.matmul(vision2, wordFeature)  # 64 196
        # sim_mapv_W = (channels ** -.5) * sim_mapv_W
        # sim_mapv_W = F.softmax(sim_mapv_W, dim=-1)
        sim_mapv_T = vision1 * TextFeature
        sim_mapv_T = F.softmax(sim_mapv_T, dim=-2)
        vision1_T = vision1 * sim_mapv_T + vision1


        # VisionFeature =VisionFeature.unsqueeze(1)
        wordFeature = wordFeature.unsqueeze(1)
        #sim_mapv_W = torch.matmul(vision2, wordFeature)  # 64 196
        #sim_mapv_W = (channels ** -.5) * sim_mapv_W
        #sim_mapv_W = F.softmax(sim_mapv_W, dim=-1)
        """
        sim_mapv_W = vision2 * wordFeature
        sim_mapv_W = F.softmax(sim_mapv_W, dim=-2)
        vision2_w  = vision2 * sim_mapv_W + vision2
        """
        vision2_w = self.crossatt(vision2, wordFeature)+ vision2

        vision = self.nn1(torch.cat([vision1_T,vision2_w],dim=2))

        out = vision.mean(1).squeeze(1)
        return out

class TextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=1024, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.attention2 = TVAttention(encoder_dim, embed_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)

        #self.decode_step = nn.LSTMCell(attention_dim+attention_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.top_down_attention = nn.LSTMCell(decoder_dim+encoder_dim+embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.language_attention = nn.LSTMCell(encoder_dim+decoder_dim, decoder_dim, bias=True)  # decoding LSTMCell

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.textencoder = TextEncoder(input_size=embed_dim, hidden_size=decoder_dim, output_size=attention_dim)
        self.nnimg = nn.Linear(encoder_dim, attention_dim)


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # 64   64
        encoder_out = encoder_out[sort_ind]

        #64 196 2048
        encoded_captions = encoded_captions[sort_ind]
        #64 52
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        embeddings1 = embeddings.clone()
        text_feature = self.textencoder(embeddings1)


        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        encoder_out_mean = encoder_out.mean(1)
        encoder_out_mean1 = encoder_out_mean.clone()
        img_feature = self.nnimg(encoder_out_mean1).squeeze(1)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            '''
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h1[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            '''

            out_feature = self.attention2(h2[:batch_size_t],  embeddings[:batch_size_t, t, :], encoder_out[:batch_size_t])

            h1, c1 = self.top_down_attention(
                torch.cat([h2[:batch_size_t], out_feature, embeddings[:batch_size_t, t, :]], dim=1),
                (h1[:batch_size_t], c1[:batch_size_t]))  # (batch_size_t, decoder_dim)
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h1[:batch_size_t])
            h2, c2 = self.language_attention(
                torch.cat([h1[:batch_size_t], attention_weighted_encoding[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind, img_feature, text_feature
