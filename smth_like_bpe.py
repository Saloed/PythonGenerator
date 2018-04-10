from collections import defaultdict


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for curr, nxt in zip(word, word[1:]):
            pairs[curr, nxt] += freq
    return pairs


def find_pair_and_replace(pair, target):
    for i in range(len(target) - 1):
        if target[i] == pair[0] and target[i + 1] == pair[1]:
            return target[:i] + (''.join(pair),) + target[i + 2:]
    return target


def merge_vocab(pair, v_in):
    v_out = {}
    for word in v_in:
        w_out = find_pair_and_replace(pair, word)
        v_out[w_out] = v_in[word]
    return v_out


def prepare_to_bpe(str_with_count):
    return {
        tuple(_str): count
        for _str, count in str_with_count
    }


def vocab_to_str_with_count(vocab):
    return [
        (''.join(word), count)
        for word, count in vocab.items()
    ]


def make_bpe(literals_with_count):
    vocab = prepare_to_bpe(literals_with_count)
    num_merges = 10000
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    vocab = list(vocab.items())
    vocab.sort(key=lambda x: x[1], reverse=True)
    return vocab  # vocab_to_str_with_count(vocab)
