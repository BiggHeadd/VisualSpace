# coding: utf-8
from tokenizers import CharBPETokenizer

# Initialize a tokenizer
merges = "./saved_tokenizer/wiki_sunyang/merges.txt"
vocab = "./saved_tokenizer/wiki_sunyang/vocab.json"
tokenizer = CharBPETokenizer(vocab, merges)

# And then encode:
encoded = tokenizer.encode("In 2012, Sun became the first Chinese man to win an Olympic gold medal in swimming.")
print(encoded.ids)
print(encoded.tokens)