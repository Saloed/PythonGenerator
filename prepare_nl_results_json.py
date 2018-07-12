import itertools
import json

from utilss import load_object, dump_object


def get_sequence(labels, probs, end_marker):
    truncated = list(itertools.takewhile(lambda x: x[0] != end_marker, zip(labels, probs)))
    labels = [l for l, _ in truncated]
    probs = [p for _, p in truncated]
    return labels, probs


def make_result(rules, words, copy, rules_prob, words_prob, rules_end, words_end):
    rules, rules_prob = get_sequence(rules, rules_prob, rules_end)
    words, words_prob = get_sequence(words, words_prob, words_end)
    words = [(w, cp) for w, cp in zip(words, copy)]
    return rules, words, rules_prob, words_prob


def main():
    result = load_object('django_xyi')
    ids, outputs, probs, targets, target_length, inputs, input_length = result

    with open('django_data_set_x') as f:
        data_set = json.load(f)
    test_set = data_set['test']

    word_seq_end = test_set['words_seq_end']
    rule_seq_end = test_set['rules_seq_end']

    result = {}
    for (rules, words, copy), (r_prob, w_prob), _id in zip(outputs, probs, ids):
        result[_id] = make_result(rules, words, copy, r_prob, w_prob, rule_seq_end, word_seq_end)

    dump_object(result, 'django_xyi_eval')


if __name__ == '__main__':
    main()
