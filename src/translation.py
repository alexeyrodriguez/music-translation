# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import librosa
import torch
from argparse import ArgumentParser
import matplotlib
import h5py
import tqdm

import utils
import wavenet_models
from utils import save_audio
from wavenet import WaveNet
from wavenet_generator import WavenetGenerator
from nv_wavenet_generator import NVWavenetGenerator


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--file', type=Path, required=True,
                        help='Input file')
    parser.add_argument('-o', '--output', type=Path,
                        help='Output file')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Checkpoint path')
    parser.add_argument('--decoders', type=int, nargs='*', default=[],
                        help='Only output for the following decoder ID')
    parser.add_argument('--rate', type=int, default=16000,
                        help='Wav sample rate in samples/second')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='Batch size during inference')
    parser.add_argument('--split-size', type=int, default=20,
                        help='Size of splits')

    return parser.parse_args()


def load_model(args):
    decoder_ids = range(2)
    checkpoints = ['{}/{}_{}.pth'.format(args.checkpoint.parent, args.checkpoint.name, i) for i in decoder_ids]
    assert len(checkpoints) >= 1, "No checkpoints found."

    model_args = torch.load(args.checkpoint.parent / 'args.pth')[0]
    encoder = wavenet_models.Encoder(model_args)
    encoder.load_state_dict(torch.load(checkpoints[0])['encoder_state'])
    encoder.eval()
    encoder = encoder.cuda()

    decoders = []
    for decoder_id, checkpoint in zip(decoder_ids, checkpoints):
        decoder = WaveNet(model_args)
        decoder.load_state_dict(torch.load(checkpoint)['decoder_state'])
        decoder.eval()
        decoder = decoder.cuda()
        decoder = WavenetGenerator(decoder, args.batch_size, wav_freq=args.rate)
        decoders.append(decoder)

    return encoder, zip(decoder_ids, decoders)


def load_wav(file_path):
    data, rate = librosa.load(file_path, sr=16000)
    assert rate == 16000
    data = utils.mu_law(data)
    return torch.tensor(data).unsqueeze(0).unsqueeze(0).float().cuda()


def save_wav(rate, file_path, audio_data):
    wav = utils.inv_mu_law(audio_data.cpu().numpy())
    save_audio(wav.squeeze(), file_path, rate=rate)


def encode(batch_size, encoder, audio_data):
    zz = []
    for xs_batch in torch.split(audio_data, batch_size):
        zz += [encoder(xs_batch)]
    zz = torch.cat(zz, dim=0)
    return zz


def decode(batch_size, split_size, decoder, zz):
    decoded = []
    for zz_batch in torch.split(zz, batch_size):
        splits = torch.split(zz_batch, split_size, -1)
        audio_data = []
        decoder.reset()
        for cond in tqdm.tqdm(splits):
            audio_data += [decoder.generate(cond).cpu()]
        audio_data = torch.cat(audio_data, -1)
        decoded.append(audio_data)
    return torch.cat(decoded, dim=0)
 

def translate(batch_size, split_size, encoder, decoders, audio_data):
    with torch.no_grad():
        zz = encode(batch_size, encoder, audio_data)
        return [(i, decode(batch_size, split_size, decoder, zz)) for i, decoder in decoders]


def main(args):
    print('Starting')
    encoder, decoders = load_model(args)
    audio_data = load_wav(args.file)[:, :, :10000]
    print(audio_data.size())

    res = translate(args.batch_size, args.split_size, encoder, decoders, audio_data)

    for i, audio_data in res:
        name = args.output.parent / f'{args.output.stem}_{i}.wav'
        save_wav(args.rate, name, audio_data)


if __name__ == '__main__':
    with torch.no_grad():
        main(parse_args())
