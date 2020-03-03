# coding:utf-8
from tokenizers import CharBPETokenizer
from pathlib import Path

# Initialize a tokenizer
tokenizer = CharBPETokenizer()

# Then train it!
tokenizer.train(["./data/wiki_sunyang.txt"])

# And you can use it
encoded = tokenizer.encode("In 2012, Sun became the first Chinese man to win an Olympic gold medal in swimming.")
# print(encoded.tokens)

# And finally save it somewhere
saved_path = Path("./saved_tokenizer/wiki_sunyang")
saved_path.mkdir(exist_ok=True, parents=True)
tokenizer.save(str(saved_path))