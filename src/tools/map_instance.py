#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#
# 生成映射后的样本
#


def main(feature_index_map,
         instance_filename,
         mapped_instance_filename,
         with_label):
    f = open(mapped_instance_filename, 'w')
    for line in open(instance_filename):
        line = line.rstrip('\r\n')
        features = line.split(' ')

        sb = ''

        # label
        if with_label:
            sb += features[0].strip()
            features = features[1:]

        # feature
        key_value_map = {}

        for feature in features:
            if len(feature) == 0:
                continue

            kv = feature.split(':')
            if len(kv) == 0:
                continue

            name = kv[0]
            if len(name) == 0:
                continue

            if len(kv) == 2:
                weight = float(kv[1])
                if weight < 1e-6:
                    continue
            else:
                weight = 1

            index = feature_index_map.get(name)
            if index is None:
                continue

            key_value_map[index] = weight

        for key in sorted(key_value_map.iterkeys()):
            sb += ' %s:%s' % (str(key), str(key_value_map[key]))
        sb = sb.strip()
        if len(sb) != 0:
            f.write(sb)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    import sys


    def usage():
        print >> sys.stderr, \
            'Usage: %s [options] FEATURE_MAP_FILENAME INSTANCE_FILENAME MAPPED_INSTANCE_FILENAME\n' \
            '  FEATURE_MAP_FILENAME: input feature map filename.\n' \
            '  INSTANCE_FILENAME: input instance filename.\n' \
            '  MAPPED_INSTANCE_FILENAME: output mapped instance filename.\n' \
            '\n' \
            '  Options:\n' \
            '    -l\n' \
            '      Whether INSTANCE_FILENAME contains labels(default is disabled).\n' % \
            sys.argv[0]
        sys.exit(1)


    if len(sys.argv) < 4:
        usage()

    with_label = 0
    i = 1
    while True:
        if sys.argv[i] == '-l':
            with_label = 1
            del sys.argv[i]
        i += 1
        if i >= len(sys.argv):
            break

    if len(sys.argv) < 4:
        usage()

    feature_index_map = {}
    index = 1
    for line in open(sys.argv[1]):
        line = line.rstrip('\r\n')
        name = line.split('\t')[0]
        feature_index_map[name] = index
        index += 1

    main(feature_index_map, sys.argv[2], sys.argv[3], with_label)
