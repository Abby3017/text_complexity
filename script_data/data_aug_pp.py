
import torch
from parrot import Parrot

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

phrases = ["ClausIE is an open information extractor, it identifies and extracts relations and their arguments in natural language text",
           "Unsupervised sentence representation learning is one of the fundamental problems in natural language processing with various downstream applications.",
           "I'm Lisa.",
           "I'm an accounting.",
           "She is a student.",
           "My favorite season is spring, my favorite time is morning, my favorite day is monday and my favorite animal is bird"]

for phrase in phrases:
    print(parrot.rephrase(input_phrase=phrase, use_gpu=True,
                          max_length=256, adequacy_threshold=0.85, do_diverse=False))
    # augment is good other arg do_diverse=True max_return_phrases=3 choose one from random
