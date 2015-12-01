#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#
# 文本匹配工具
#

import re


def string_2_unicode(content):
    """
    尝试各种方式解码content, 返回以unicode编码的字符串, 解码失败返回None.

    尝试的解码方式:
    ANSI格式编码
    GB18030(完全兼容GB2312)
    GBK
    UTF-8 无BOM格式编码
    UTF-8 格式编码
    UCS-2 Big Endian 格式编码
    UCS-2 Little Endian 格式编码
    """
    UTF8_BOM = '\xef\xbb\xbf'
    UTF16_BE_BOM = '\xfe\xff'
    UTF16_LE_BOM = '\xff\xfe'
    CHARSET_LIST = ['utf-8', 'gb18030', 'gbk']

    if content.startswith(UTF8_BOM):
        content = content[3:].decode('utf-8')
    elif content.startswith(UTF16_BE_BOM):
        content = content[2:].decode('utf-16-be')
    elif content.startswith(UTF16_LE_BOM):
        content = content[2:].decode('utf-16-le')
    else:
        charset = None
        for cs in CHARSET_LIST:
            try:
                content = content.decode(cs)
                charset = cs
                break
            except Exception:
                pass
        if charset is None:
            # 解码失败
            return None
    return content


class ACMatch:
    """
    AC多模式匹配引擎. "包含"匹配.
    """

    def __init__(self, kw_list, case_sensitive=False):
        """
        kw_list: 关键词(已转换为unicode)
        case_sensitive: 大小写敏感
        """
        assert len(kw_list) >= 1
        # 多个pattern组成的list
        self.patterns = set()
        for kw in kw_list:
            if case_sensitive:
                self.patterns.add(kw)
            else:
                self.patterns.add(kw.lower())
        # 状态转移
        self.goto = {}
        # 失败转移
        self.failure = {}
        # 匹配输出
        self.output = {}
        self.case_sensitive = case_sensitive

        self.__construct_goto()
        self.__construct_failure()

    def __del__(self):
        self.patterns = None
        self.goto = None
        self.failure = None
        self.output = None

    def __construct_goto(self):
        """
        构造状态转移数据结构和匹配输出中间状态
        """
        # 全局唯一状态编号
        global_state = 1
        # 遍历每一个pattern
        for item in self.patterns:
            # 从状态0开始
            new_state = 0
            # 从pattern的第一个字符开始
            index = 0
            # 复用前面已经构造好的前缀字符串
            while new_state in self.goto and index < len(item) \
                    and item[index] in self.goto[new_state]:
                new_state = self.goto[new_state][item[index]]
                index += 1
            for i in range(index, len(item)):
                self.goto.setdefault(new_state, {})
                self.goto[new_state][item[i]] = global_state
                new_state = global_state
                global_state += 1

            # 构造匹配输出的中间状态
            self.output.setdefault(new_state, [])
            self.output[new_state].append(item)

    def __construct_failure(self):
        """
        构造失败转移数据结构和匹配输出最终结构
        """
        # 图的宽度优先遍历
        queue_list = []

        # 初始化深度为1的结点
        for key in self.goto[0]:
            state_depth_1 = self.goto[0][key]
            self.failure[state_depth_1] = 0
            queue_list.append(state_depth_1)

        # 宽度优先遍历
        while len(queue_list) > 0:
            # queue pop
            state = queue_list[0]
            queue_list = queue_list[1:]
            if state in self.goto:
                for key in self.goto[state]:
                    state_depth_n = self.goto[state][key]
                    queue_list.append(state_depth_n)
                    temp_state = self.failure[state]
                    while (temp_state not in self.goto or key not in self.goto[
                        temp_state]) \
                            and temp_state != 0:
                        temp_state = self.failure[temp_state]
                    # 找到公共前缀
                    if temp_state in self.goto and key in self.goto[temp_state]:
                        self.failure[state_depth_n] = self.goto[temp_state][key]
                    else:
                        self.failure[state_depth_n] = 0

                    if self.failure[state_depth_n] in self.output:
                        self.output.setdefault(state_depth_n, [])
                        self.output[state_depth_n].extend(
                            self.output[self.failure[state_depth_n]])

    def match(self, text):
        if not self.case_sensitive:
            text = text.lower()
        matches = []
        # 当前的状态编号
        state = 0
        # 遍历文本的每一个字符
        for item in text:
            # 如果正常转移不下去，就进行失败转移过程
            while (state not in self.goto or item not in self.goto[state]) \
                    and state != 0:
                state = self.failure[state]

            # 进行正常转移
            if state in self.goto and item in self.goto[state]:
                state = self.goto[state][item]
            else:
                state = 0

            if state in self.output:
                # # 只输出最长的匹配串
                # max_len = 0
                # max_len_output = ''
                # for output in self.output[state]:
                #     if max_len < len(output):
                #         max_len = len(output)
                #         max_len_output = output
                # if len(matches) != 0 and max_len_output.startswith(matches[-1]):
                #     matches[-1] = max_len_output
                # else:
                #     matches.append(max_len_output)
                for output in self.output[state]:
                    matches.append(output)
        return matches


class ReMatch:
    """
    正则表达式匹配引擎. "单词"匹配.
    """

    SPACE = re.compile(r'\s+')

    def __process_kw(self, kw):
        """
        将输入的kw转换为正则表达式的形式
        """
        if not self.case_sensitive:
            kw = kw.lower()
        kw = ReMatch.SPACE.sub('\s+', kw)
        return r'%s' % kw

    def __process_matched_kw(self, kw):
        """
        将匹配到的文本转换为输入的kw
        """
        if not self.case_sensitive:
            kw = kw.lower()
        kw = ReMatch.SPACE.sub(' ', kw)
        return kw

    def __init__(self, kw_list, case_sensitive=False):
        assert len(kw_list) >= 1

        self.kw_list = kw_list
        self.case_sensitive = case_sensitive
        re_str = r'\b('
        re_str += self.__process_kw(kw_list[0])
        for kw in kw_list[1:]:
            re_str += r'|' + self.__process_kw(kw)
        re_str += r')\b'

        if case_sensitive:
            self.pattern = re.compile(re_str)
        else:
            self.pattern = re.compile(re_str, re.IGNORECASE)

    def match(self, text):
        matched_kw_list = self.pattern.findall(text)
        return [self.__process_matched_kw(kw) for kw in matched_kw_list]


class KWMatch:
    """
    对ACMatch和ReMatch的一个封装:
    对非"全英文数字"的关键词采用"包含"匹配.
    对"全英文数字"的关键词采用"单词"匹配. 例如: 关键词"ap"不能匹配到"app"和"apply".
    """

    DIGIT_LETTER = re.compile(r'^[A-Za-z0-9 ]+$')

    def __init__(self, kw_list, case_sensitive=False):
        ac_kw_list = []
        re_kw_list = []
        for kw in kw_list:
            if KWMatch.DIGIT_LETTER.match(kw):
                re_kw_list.append(kw)
            else:
                ac_kw_list.append(kw)

        if len(ac_kw_list) == 0:
            self.ac_match = None
        else:
            self.ac_match = ACMatch(ac_kw_list, case_sensitive)

        if len(re_kw_list) == 0:
            self.re_match = None
        else:
            self.re_match = ReMatch(re_kw_list, case_sensitive)

    def match_set(self, text):
        matched = []
        if self.ac_match is not None:
            matched += self.ac_match.match(text)
        if self.re_match is not None:
            matched += self.re_match.match(text)
        return set(matched)

    def match(self, text):
        return list(self.match_set(text))


class KWMatch_And:
    """
    基于KWMatch的基础上, 支持kw中带有空格, 空格分隔开的sub kw之间用"与"进行匹配.
    """
    DIGIT_LETTER = re.compile(r'^[A-Za-z0-9 ]+$')

    def __init__(self, kw_list, case_sensitive=False):
        self.simple_kw = set()
        self.sub_kw_to_sub_kw_set = {}
        atomic_kw_list = set()

        for kw in kw_list:
            if ' ' in kw:
                sub_kw_set = set()
                for sub_kw in kw.split(' '):
                    if len(sub_kw) != 0:
                        sub_kw_set.add(sub_kw)
                        atomic_kw_list.add(sub_kw)
                for sub_kw in sub_kw_set:
                    self.sub_kw_to_sub_kw_set.setdefault(sub_kw, [])
                    self.sub_kw_to_sub_kw_set[sub_kw].append(sub_kw_set)
            else:
                self.simple_kw.add(kw)
                atomic_kw_list.add(kw)

        self.kw_match = KWMatch(list(atomic_kw_list), case_sensitive)

    def match_set(self, text):
        ret = set()
        matched = self.kw_match.match_set(text)

        for kw in matched:
            if kw in self.simple_kw:
                ret.add(kw)

            sub_kw_set_list = self.sub_kw_to_sub_kw_set.get(kw)
            if sub_kw_set_list is not None:
                for sub_kw_set in sub_kw_set_list:
                    match_flag = True
                    for sub_kw in sub_kw_set:
                        if sub_kw not in matched:
                            match_flag = False
                            break
                    if match_flag:
                        ret.add(' '.join(sub_kw_set))

        return ret

    def match(self, text):
        return list(self.match_set(text))


def load_lines(kw_filename):
    """
    加载文件, 并转换为unicode编码.
    返回一个list, 每个元素对应文件中的一行.
    """
    f = open(kw_filename, 'r')
    content = f.read()
    f.close()

    content = string_2_unicode(content)

    kw_list = []
    for line in content.split('\n'):
        line = line.strip('\r').strip()
        if len(line) == 0:
            continue
        kw_list.append(line)
    return kw_list


def test_ac_match(kw_filename, article_filename):
    kw_list = load_lines(kw_filename)
    ac_match = ACMatch(kw_list, True)
    for a in load_lines(article_filename):
        matches = ac_match.match(a)
        print a
        for match in matches:
            print 'keyword: "%s"' % match
        print ''


g_ac_match = None


def match_keyword_ac_match(query, kw_filename, case_sensitive=False):
    global g_ac_match
    if g_ac_match is None:
        kw_list = load_lines(kw_filename)
        g_ac_match = ACMatch(kw_list, case_sensitive)

    if query is None:
        return 0

    try:
        matches = g_ac_match.match(query)
    except:
        return 0

    if len(matches) != 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    ac_match = ACMatch(
        ['中国', '美国', '大众', '中国美国', '美国大众', '中国美国大众',
         'ap', 'app', 'apply', 'App', 'c++'], True)
    assert ac_match.match('apply') == ['ap', 'app', 'apply']
    assert ac_match.match('app') == ['ap', 'app']
    assert ac_match.match('App') == ['App']
    assert ac_match.match('中国美国') == ['中国', '中国美国', '美国']
    assert ac_match.match('java c++') == ['c++']

    re_match = ReMatch(
        ['kw1', 'kw2', 'AP', 'aPp', 'aPPly', 'cet-6', 'cet 6'], False)
    assert re_match.match('kw1') == ['kw1']
    assert re_match.match(' kw1') == ['kw1']
    assert re_match.match('kw1 ') == ['kw1']
    assert re_match.match(' kw2 ') == ['kw2']
    assert re_match.match('汉字kw1') == ['kw1']
    assert re_match.match('kw1汉字') == ['kw1']
    assert re_match.match('汉字ap汉字') == ['ap']
    assert re_match.match('kw1 kw2') == ['kw1', 'kw2']
    assert re_match.match('an android app') == ['app']
    assert re_match.match('cet-6 in China') == ['cet-6']
    assert re_match.match('cEt 6 in China') == ['cet 6']
    assert re_match.match('cet  6 in China') == ['cet 6']
    assert re_match.match('cet\t6 in China') == ['cet 6']
    assert re_match.match('中国cet 6') == ['cet 6']
    assert re_match.match('cet 6中国') == ['cet 6']
    assert re_match.match('app apply') == ['app', 'apply']
    assert re_match.match('kw1s') == []
    assert re_match.match('kw1kw2') == []
    assert re_match.match('cet\t6x') == []
    assert re_match.match('0cet 6') == []

    kw_match = KWMatch(
        ['中国', '美国', '大众', 'kw1', 'kw2', 'ap', 'app', 'apply', 'cet 6', 'c++'],
        False)
    assert kw_match.match('中国app美国大众') == ['中国', '美国', '大众', 'app']
    assert kw_match.match('中国apply') == ['中国', 'apply']
    assert kw_match.match('中国cet   6') == ['中国', 'cet 6']
    assert kw_match.match('中国1app') == ['中国']
    assert kw_match.match('java c++') == ['c++']

    kw_match = KWMatch_And(['我  你', '美国', '中国', '中国 美国'], False)
    assert kw_match.match('中国美国') == ['中国', '美国', '中国 美国']
    assert kw_match.match('美国 中国') == ['中国', '美国', '中国 美国']
    assert kw_match.match('美国中国') == ['中国', '美国', '中国 美国']
    assert kw_match.match('中国人民') == ['中国']
