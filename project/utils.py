import torch as th


word2idx = {
    "<SOS>": 0,
    "<EOS>": 1,
    "<PAD>": 2,
    "0": 3,
    "1": 4,
    "2": 5,
    "3": 6,
    "4": 7,
    "5": 8,
    "6": 9,
    "7": 10,
    "8": 11,
    "9": 12,
    "T": 13,
}


idx2word = {
    0: "<SOS>",
    1: "<EOS>",
    2: "<PAD>",
    3: "0",
    4: "1",
    5: "2",
    6: "3",
    7: "4",
    8: "5",
    9: "6",
    10: "7",
    11: "8",
    12: "9",
    13: "T",
}


def text_to_tensor(text: str):
    seq = [word2idx["<SOS>"]]
    for char in text:
        if char in word2idx:
            seq.append(word2idx[char])
    
    seq.append(word2idx["<EOS>"])

    return th.tensor(seq).long()
