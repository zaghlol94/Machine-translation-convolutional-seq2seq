import torch
import pickle
import spacy
import argparse
from utils import translate_sentence, display_attention
from config import config
from model import Encoder, Decoder, Seq2Seq

parser = argparse.ArgumentParser(description="translate string from german to english")
parser.add_argument("-s", "--sentence", type=str, required=True, help="sentence that you want to translate")
args = parser.parse_args()

print(args.sentence)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_tokenizer = spacy.load('de_core_news_sm')
trg_tokenizer = spacy.load('en_core_web_sm')

with open('src_vocab.pkl', 'rb') as inp:
    src_vocab = pickle.load(inp)
with open('trg_vocab.pkl', 'rb') as inp:
    trg_vocab = pickle.load(inp)

TRG_PAD_IDX = trg_vocab.stoi["<pad>"]

INPUT_DIM = len(src_vocab.stoi)
OUTPUT_DIM = len(trg_vocab.stoi)

enc = Encoder(INPUT_DIM, config["EMB_DIM"], config["HID_DIM"], config["ENC_LAYERS"], config["ENC_KERNEL_SIZE"], config["ENC_DROPOUT"], device)
dec = Decoder(OUTPUT_DIM, config["EMB_DIM"], config["HID_DIM"], config["DEC_LAYERS"], config["DEC_KERNEL_SIZE"], config["DEC_DROPOUT"], TRG_PAD_IDX, device)

model = Seq2Seq(enc, dec).to(device)
model.load_state_dict(torch.load(config["test_config"]["model_path"]))
translation, attention = translate_sentence(
        args.sentence, src_vocab, trg_vocab, src_tokenizer, model, device
    )
print(" ".join(translation[:-1]))
display_attention(args.sentence.split(" "), translation, attention)

