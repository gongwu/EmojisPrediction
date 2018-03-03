import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(jsonfile):
    ROOT = '../data/twitter_data/English'
    DATA_DIR = ROOT + '/word'
    OUTPUT_DIR = '../output'
    VOCABULARY_DIR = '../vocabulary'
    MODEL_DIR = '../model'
    DIC_DIR = '../dic'
    config, _ = get_config_from_json(jsonfile)
    config.summary_dir = os.path.join("../experiments", config.exp_name, "summary")
    config.train_file = os.path.join(ROOT, "processed", "2of3.json")
    config.dev_file = os.path.join(ROOT, "processed", "1of3.json")
    config.test_file = None
    config.word_embed_file = os.path.join(ROOT, "embed", "SWM.vocab.vector")
    config.w2i_file = os.path.join(DIC_DIR, "w2i.p")
    config.c2i_file = os.path.join(DIC_DIR, "c2i.p")
    config.n2i_file = os.path.join(DIC_DIR, "n2i.p")
    config.p2i_file = os.path.join(DIC_DIR, "p2i.p")
    config.oov_file = os.path.join(DIC_DIR, "oov.p")
    config.we_file = os.path.join(DIC_DIR, "we.p")
    config.dev_model_file = os.path.join(MODEL_DIR, "dev_model")
    config.VOCAB_NORMAL_WORDS_PATH = os.path.join(VOCABULARY_DIR, "normal_word.pkl")
    config.max_sent_len = 34
    config.max_word_len = 38
    config.num_class = 20
    config.word_dim = 300
    config.char_dim = 50
    config.ner_dim = 50
    config.pos_dim = 50

    return config