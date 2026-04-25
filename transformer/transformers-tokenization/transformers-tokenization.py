import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE

        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i 
            self.id_to_word[i] = token 

        vocab_size = 4
        unique_words = set() 

        for text in texts:
            text = text.lower().split()
            unique_words.update(text)

        unique_words = sorted(unique_words) 
        for word in unique_words:
            self.word_to_id[word] = vocab_size
            self.id_to_word[vocab_size] = word
            vocab_size += 1
            
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        text = text.lower().split()
        unk_id = self.word_to_id.get(self.unk_token, 1)
        encode = [self.word_to_id.get(x, unk_id) for x in text]
        return encode 
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE

        decode = [self.id_to_word.get(id, self.unk_token) for id in ids]
        return " ".join(decode)