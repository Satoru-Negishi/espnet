from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from espnet2.tts.gst.style_encoder import StyleEncoder
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram


import argparse
from pathlib import Path
import os
from espnet2.fileio.sound_scp import SoundScpReader
import numpy as np
from tqdm import tqdm
import sys
import csv

from pprint import pprint
import soundfile
"""
プログラムの大枠
Reference Code:
    /espnet/egs2/TEMPLATE/tts1/pyscripts/utils/extract_xvectors.py

"""


# feats_extractor_choices = ClassChoices(
#     "feats_extract",
#     classes=dict(
#         fbank=LogMelFbank,
#         spectrogram=LogSpectrogram,
#         linear_spectrogram=LinearSpectrogram,
#     ),
#     type_check=AbsFeatsExtract,
#     default="fbank",
# )


def get_parser():
    """Construct the parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "in_folder", type=Path, help="Path to the input kaldi data directory."
    )
    parser.add_argument(
        "out_folder",
        type=Path,
        help="Output folder to save the style embedding.",
    )
    return parser

from typing import Any


class GST(object):
    def __init__(
        self,
        # idim: int,
        odim: int,
        feats_extract: Optional[AbsFeatsExtract],
        adim: int = 256,
        gst_tokens: int = 10,
        gst_heads: int = 4,
        gst_conv_layers: int = 6,
        gst_conv_chans_list: Sequence[int] = (128, 128, 256, 256, 512, 512),
        gst_conv_kernel_size: int = 3,
        gst_conv_stride: int = 2,
        gst_gru_layers: int = 1,
        gst_gru_units: int = 128,
    ):
        """GST model.
        
        Reference Code:
            /mnt/data/users/snegishi/M2/Satoru-Negishi/espnet/espnet2/tts/fastspeech2/fastspeech2.py  ,321~
        """
        self.gst = StyleEncoder(
            idim=odim,  # the input is mel-spectrogram
            gst_tokens=gst_tokens,
            gst_token_dim=adim,
            gst_heads=gst_heads,
            conv_layers=gst_conv_layers,
            conv_chans_list=gst_conv_chans_list,
            conv_kernel_size=gst_conv_kernel_size,
            conv_stride=gst_conv_stride,
            gru_layers=gst_gru_layers,
            gru_units=gst_gru_units,
        )
        self.feats_extract = feats_extract

    def extract_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """Extract features.

        Args:
            speech (Tensor): Input speech feature (T, D).
            speech_lengths (Tensor): The length of input speech feature (N,).

        Reference Code:
            /mnt/data/users/snegishi/M2/Satoru-Negishi/espnet/espnet2/tts/espnet_model.py  ,153~
        """
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(speech, speech_lengths)
        else:
            # Use precalculated feats (feats_type != raw case)
            feats, feats_lengths = speech, speech_lengths
        feats_dict = dict(feats=feats, feats_lengths=feats_lengths)

        return feats_dict

    def __call__(self, speech, speech_lengths):
        embeds = self.extract_feats(speech, speech_lengths)
        return embeds
    
def sound_reader(rate, audio):
    speech = np.expand_dims(audio, 0)
    speech_lengths = np.array([speech.shape[0]])
    return speech, speech_lengths

def main(argv):
    """Load the model, generate kernel and bandpass plots."""
    parser = get_parser()
    args = parser.parse_args(argv)

    # if torch.cuda.is_available() and ("cuda" in args.device):
    #     device = args.device
    # else:
    #     device = "cpu"

    # Prepare spk2utt for mean x-vector
    spk2utt = dict()
    with open(os.path.join(args.in_folder, "spk2utt"), "r") as reader:
        for line in reader:
            details = line.split()
            spk2utt[details[0]] = details[1:]

    wav_scp = SoundScpReader(os.path.join(args.in_folder, "wav.scp"), np.float32)
    # os.makedirs(args.out_folder, exist_ok=True)
    # writer_utt = kaldiio.WriteHelper(
    #     "ark,scp:{0}/xvector.ark,{0}/xvector.scp".format(args.out_folder)
    # )
    # writer_spk = kaldiio.WriteHelper(
    #     "ark,scp:{0}/spk_xvector.ark,{0}/spk_xvector.scp".format(args.out_folder)
    # )
    writer_spk = {}
    #feats_extract
    """
    Reference Code:
        /espnet/espnet2/tasks/tts.py 295~ 
    """
    # 1. feats_extract
    feats_extract_class = LinearSpectrogram()
    # feats_extract = feats_extract_class(**args.feats_extract_conf)
    odim = feats_extract_class.output_size()

    gst_encoder = GST(odim=odim,feats_extract=feats_extract_class)
    count = 0
    for speaker in tqdm(spk2utt):
        style_embeds = []
        for utt in spk2utt[speaker]:            
            rate, audio = wav_scp[utt]
            speech, speech_lengths = sound_reader(rate, audio)
            # Style Embedding
            embeds = gst_encoder(torch.tensor(speech), torch.tensor(speech_lengths))
            # pprint(embeds["feats"][0][0])
            print(embeds["feats"].shape)
            # # writer_utt[utt] = np.squeeze(embeds)
            # style_embeds.append(embeds['feats'].tolist())
            # break
        sys.exit(0)
        # if count == 1:
        #     sys.exit(0)
        # count += 1
        # Speaker Normalization
        # print(style_embeds[0][0][1])
        #TODO:
        # -style_emeadsのサイズの問題によりnp.meanができない問題発生中，次回修正
        # stack_embeds = np.stack(style_embeds, 0)
        # print(stack_embeds)
        # embeds = np.mean(np.stack(style_embeds, 0), 0)
        # writer_spk[speaker] = embeds

    # writer_utt.close()
    # writer_spk.close()
    with open("/mnt/data/users/snegishi/M2/Satoru-Negishi/espnet/egs2/jvs_promptttspp/writer_spk.csv", "w") as f:
        writer = csv.writer(f)
        for k, v in writer_spk.items():
            writer.writerow([k, v])
    
if __name__ == "__main__":
    main(sys.argv[1:])