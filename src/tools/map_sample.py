#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Yafei Zhang (zhangyafeikimi@gmail.com)
#
# map non-LIBSVM sample files to LIBSVM format with a feature map
#

import re


def main(feature_index_map, sample_filename_list, with_label):
    splitter = re.compile(r'( |\t|\|)+')

    for sample_filename in sample_filename_list:
        print 'Mapping "%s" to "%s.libsvm"...' % (
            sample_filename, sample_filename)
        f = open(sample_filename + '.libsvm', 'w')
        for line in open(sample_filename):
            line = line.rstrip('\r\n')
            features = splitter.split(line)
            features = [feature for feature in features if
                        feature not in ('', ' ', '\t', '|')]

            if len(features) != 0:
                sb = ''

                # label
                if with_label != 0:
                    sb += features[0]
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
                        if -1e-6 < weight < 1e-6:
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

    feature_map_filename = 'feature-map'
    with_label = 1


    def usage():
        print >> sys.stderr, \
            'Usage: %s [options] SAMPLE_FILE1 [SAMPLE_FILE2] ...\n' \
            '  SAMPLE_FILE: input sample filename.\n' \
            '    A postfix ".libsvm" will be added to SAMPLE_FILE.\n' \
            '\n' \
            '  Options:\n' \
            '    -f FEATURE_MAP_FILENAME\n' \
            '      The input feature map filename.\n' \
            '      Default is "%s".\n' \
            '    -l WITH_LABEL(0 or 1)\n' \
            '      Whether SAMPLE_FILE contains labels.\n' \
            '      Default is "%d".\n' % \
            (sys.argv[0], feature_map_filename, with_label)
        sys.exit(1)


    if len(sys.argv) == 1:
        usage()

    i = 1
    while True:
        if sys.argv[i] in ('-h', '-help', '--help'):
            usage()

        if sys.argv[i] == '-f':
            if i + 1 == len(sys.argv):
                print >> sys.stderr, '"%s" wants a value' % sys.argv[i]
                usage()
            feature_map_filename = sys.argv[i + 1]
            del sys.argv[i:i + 2]
        elif sys.argv[i] == '-l':
            if i + 1 == len(sys.argv):
                print >> sys.stderr, '"%s" wants a value' % sys.argv[i]
                usage()
            with_label = int(sys.argv[i + 1])
            del sys.argv[i:i + 2]
        else:
            i += 1
        if i == len(sys.argv):
            break

    if len(sys.argv) == 1:
        usage()

    feature_index_map = {}
    index = 1
    for line in open(feature_map_filename):
        line = line.rstrip('\r\n')
        name = line.split('\t')[0]
        feature_index_map[name] = index
        index += 1

    sample_filename_list = sys.argv[1:]
    main(feature_index_map, sample_filename_list, with_label)
