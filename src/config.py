config = {
    "src_train": "../data-set/train.de",
    "trg_train":  "../data-set/train.en",
    "src_valid": "../data-set/val.de",
    "trg_valid":  "../data-set/val.en",
    "EMB_DIM": 256,
    "HID_DIM": 512,
    "ENC_LAYERS": 10, # number of conv. blocks in encoder
    "DEC_LAYERS": 10, # number of conv. blocks in decoder
    "ENC_KERNEL_SIZE": 3, # must be odd!
    "DEC_KERNEL_SIZE": 3, # can be even or odd
    "ENC_DROPOUT": 0.25,
    "DEC_DROPOUT":  0.25,
    "N_EPOCHS":  10,
    "CLIP": 1,
    "learning_rate": 0.001,
    "test_config": {
        "model_path": "model.pt",
        "src_test": "../data-set/test_2016_flickr.de",
        "trg_test": "../data-set/test_2016_flickr.en",
    }
}
