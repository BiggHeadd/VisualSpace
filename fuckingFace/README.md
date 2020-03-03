## Tokenizer

- 训练一个分词器或者直接读取训练好的分词器
    - 依赖文件
        - merges.txt
        - vocab.json
    
    - 训练新的分词器
        - 初始化分词器
            - tokenizer = CharBPETokenizer()
        - 训练
            - tokenizer.train(["./data/wiki_sunyang.txt"])
        - 完成
    - 读取已有分词器
        - 读取
            - tokenizer = CharBPETokenizer(vocab, merges)
        - 完成

- 建立自己的分词器
    - 从现有的分词器改造
        - 读取分词模型
            - bpe = models.BPE.from_files(vocab, merges)
        - 初始化分词器
            - tokenizer = Tokenizer(bpe)
        - 定制分词器和解码器
            - tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            - tokenizer.decoder = decoders.ByteLevel()
        - 完成
    - 训练一个分词器
        - 初始化一个空的分词器
            - tokenizer = Tokenizer(models.BPE.empty())
        - 定制分词器和解码器
            - tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            - tokenizer.decoder = decoders.ByteLevel()
        - 训练
            - trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2)
            - tokenizer.train(trainer, ["path/.txt"])
        - 完成