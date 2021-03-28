import argparse

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emb_f", help="path to pre-trained fastext embeddings (*.vec file)")
    parser.add_argument("out_dir", help="path to save trained model")
    parser.add_argument("-learning_rate", default=1e-3, type=float, help='learning rate [default: 0.001]')
    parser.add_argument("-epochs", default=10, type=int, help='number of training epochs [default: 10]')
    parser.add_argument("-dropout", default=0.25, type=float, help='the probability for dropout [default: 0.25]')
    parser.add_argument("-embedding_size", default=300, type=int, help='number of embedding dimension [default: 300]')
    parser.add_argument("-max_seq_len", default=100, type=int, help='maximum sequence length [default: 100]')
    parser.add_argument("-batch_size", default=64, type=int, help='batch size while training [default: 64]')
    parser.add_argument("-save_model", default=True, type=bool, help='save model  [default: True]')
    args = parser.parse_args()
    return args