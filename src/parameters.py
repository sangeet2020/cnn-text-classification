from dataclasses import dataclass
from arguments import parse_arguments

args = parse_arguments()

@dataclass
class Parameters:
    # Preprocessing parameeters
    
    max_seq_len: int = args.max_seq_len
    num_words: int = 179209

    # Model parameters
    embedding_size: int = args.embedding_size
    out_size: int = 32
    stride: int = 2
    dropout: float = args.dropout

    # Training parameters
    epochs: int = args.epochs
    batch_size: int = args.batch_size
    
    learning_rate: float = args.learning_rate
