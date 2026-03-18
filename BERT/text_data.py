import collections
import re
import jieba

def s():
    with open("thuc_no.txt", 'r', encoding='utf-8') as f:
        lines = f.read()
    pattern = r'[^a-zA-Z0-9\u4e00-\u9fa5]'  # \u4e00-\u9fa5 是中文字符的Unicode范围
    cleaned_text = re.sub(pattern, '', lines)
    return cleaned_text

def tokenize(lines, token='word'):
    if token == 'word':
        #return list(jieba.cut(token))
         return [list(jieba.cut(line)) if any('\u4e00' <= char <= '\u9fff' for char in line)
                 else line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print("Error:未知令牌类型：" + token)

def count_corpus(tokens):
    """统计标记的频率：这里的tokens是1D列表或者2D列表"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将tokens展平成使用标记填充的一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    @property
    def unk(self):
        return 0
    @property
    def token_freqs(self):
        return self._token_freqs