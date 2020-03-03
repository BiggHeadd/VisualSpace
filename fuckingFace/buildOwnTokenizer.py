# coding:utf-8
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

print("训练一个新的分词器")
"""
Train a new tokenizer
训练一个新的分词器
"""

# Initialize a tokenizer
# 初始化一个分词器
tokenizer = Tokenizer(models.BPE.empty())

# Customize pre-tokenization and decoding
#　定制预训练分词器和解码器
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

# And then train
# 训练
trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2)
tokenizer.train(trainer, [
    "./data/wiki_sunyang.txt"
])

# Now we can encode
#　分词
encoded = tokenizer.encode("In 2012, Sun became the first Chinese man to win an Olympic gold medal in swimming.")
print(encoded.ids)
print(encoded.tokens)

print("\n使用一个预训练的分词器")

"""
Use a pre-trained tokenizer
使用一个预训练的分词器
"""
# Load a pre-trained tokenizer
# 读取一个预训练的分词器
merges = "./saved_tokenizer/wiki_sunyang/merges.txt"
vocab = "./saved_tokenizer/wiki_sunyang/vocab.json"
bpe = models.BPE.from_files(vocab, merges)

# Initialize a tokenizer
# 初始化一个分词器
tokenizer = Tokenizer(bpe)

# Customize pre-tokenization and decoding
# 定制一个预训练分词器和解码器
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

# And then encode
# 然后就可以编码了
encoded = tokenizer.encode("In 2012, Sun became the first Chinese man to win an Olympic gold medal in swimming.")
print(encoded.ids)
print(encoded.tokens)

# Or tokenize multiple sentences at once:
# 可以一次性编码一批句子
encoded = tokenizer.encode_batch([
    "In 2012, Sun became the first Chinese man to win an Olympic gold medal in swimming.",
    "In 2012, Sun became the first Chinese man to win an Olympic gold medal in swimming."
])
print(encoded)