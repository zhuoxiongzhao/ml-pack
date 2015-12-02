#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#
# extract ngram tokens and their frequency
#


class NGram:
    def __init__(self, _min, _max):
        self.length = None
        self._min = _min
        self._max = _max
        self.table = {}

    def _parse_text(self, text, n):
        chars = ' ' * n
        for letter in (" ".join(text.split()) + " "):
            chars = chars[1:] + letter
            self.table[chars] = self.table.get(chars, 0) + 1

    def parse_text(self, text):
        for i in range(self._min, self._max + 1):
            self._parse_text(text, i)


def prepare_4_ngram(text):
    CHARSET_LIST = ['utf-8', 'gb18030', 'gbk']
    charset = None
    for cs in CHARSET_LIST:
        try:
            text = text.decode(cs)
            charset = cs
            break
        except Exception:
            pass
    if charset is None:
        # decode failed
        return None

    sb = unicode('')
    for uchar in text:
        if u'\u4e00' <= uchar <= u'\u9fa5':
            # Chinese characters
            sb += uchar
        if u'\u0030' <= uchar <= u'\u0039':
            # digits
            sb += uchar
        if (u'\u0041' <= uchar <= u'\u005a') \
                or (u'\u0061' <= uchar <= u'\u007a'):
            # letters
            sb += uchar.lower()
    return sb


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:
        _min = int(sys.argv[1])
    else:
        _min = 2
    if len(sys.argv) > 3:
        _max = int(sys.argv[2])
    else:
        _max = 5
    if len(sys.argv) > 4:
        count_threshold = int(sys.argv[3])
    else:
        count_threshold = 100

    ngram = NGram(_min, _max)
    for line in sys.stdin:
        text = prepare_4_ngram(line)
        ngram.parse_text(text)

    ngram_by_count = []
    for key, value in ngram.table.iteritems():
        ngram_by_count.append((value, key))
    ngram_by_count = sorted(ngram_by_count, reverse=True)
    for count, token in ngram_by_count:
        if count > count_threshold:
            print count, token.encode('utf-8')
