#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#
# 生成特征映射表
#


def main(instance_filename, feature_map_filename, threshold):
    feature_count_map = {}
    for line in open(instance_filename):
        line = line.rstrip('\r\n')
        features = line.split(' ')

        for feature in features[1:]:
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

            if name not in feature_count_map:
                feature_count_map[name] = 1
            else:
                count = feature_count_map[name]
                feature_count_map[name] = count + 1

    # 保留频率大于某阈值的特征
    ffeature_map = open(feature_map_filename, 'w')
    for name, count in feature_count_map.items():
        if count > threshold:
            ffeature_map.write(name)
            ffeature_map.write('\t')
            ffeature_map.write(str(count))
            ffeature_map.write('\n')
    ffeature_map.close()


if __name__ == '__main__':
    import sys


    def usage():
        print >> sys.stderr, \
            'Usage: %s [options] INSTANCE_FILENAME FEATURE_MAP_FILENAME\n' \
            '  INSTANCE_FILENAME: input instance filename.\n' \
            '  FEATURE_MAP_FILENAME: output feature map filename.\n' \
            '\n' \
            '  Options:\n' \
            '    -f THRESHOLD\n' \
            '      Keep features whose frequency are larger than this threshold.\n' % \
            sys.argv[0]
        sys.exit(1)


    if len(sys.argv) < 3:
        usage()

    threshold = 10
    i = 1
    while True:
        if sys.argv[i] == '-f':
            threshold = int(sys.argv[i + 1])
            del sys.argv[i:i + 2]
        i += 2
        if i >= len(sys.argv):
            break

    if len(sys.argv) < 3:
        usage()

    main(sys.argv[1], sys.argv[2], threshold)
