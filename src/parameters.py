from dataclasses import dataclass

@dataclass
class Parameters:
    # Preprocessing parameeters
    max_seq_len: int = 100
    num_words: int = 2000

    # Model parameters
    embedding_size: int = 300
    out_size: int = 32
    stride: int = 2

    # Training parameters
    epochs: int = 10
    batch_size: int = 64
    
    learning_rate: float = 0.001
