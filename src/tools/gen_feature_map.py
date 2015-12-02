#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#
# generate feature map from non-LIBSVM format sample files
#

import re

feature_count_map = {}


def main(sample_filename_list, with_label):
    splitter = re.compile(r'( |\t|\|)+')

    for sample_filename in sample_filename_list:
        print 'Processing "%s"...' % sample_filename
        for line in open(sample_filename):
            line = line.rstrip('\r\n')
            features = splitter.split(line)
            features = [feature for feature in features if
                        feature not in ('', ' ', '\t', '|')]

            # label
            if with_label != 0:
                features = features[1:]

            # feature
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

                if name not in feature_count_map:
                    feature_count_map[name] = 1
                else:
                    count = feature_count_map[name]
                    feature_count_map[name] = count + 1


def save_feature_map(feature_map_filename, threshold):
    print 'Writing to "%s"...' % feature_map_filename
    f = open(feature_map_filename, 'w')
    for name, count in feature_count_map.items():
        if count > threshold:  # filter by threshold
            f.write(name)
            f.write('\t')
            f.write(str(count))
            f.write('\n')
    f.close()
    print 'Done.'


if __name__ == '__main__':
    import sys

    feature_map_filename = 'feature-map'
    with_label = 1
    threshold = 0


    def usage():
        print >> sys.stderr, \
            'Usage: %s [options] SAMPLE_FILE1 [SAMPLE_FILE2] ...\n' \
            '  SAMPLE_FILE: input sample filename.\n' \
            '\n' \
            '  Options:\n' \
            '    -f FEATURE_MAP_FILENAME\n' \
            '      The output feature map filename.\n' \
            '      Default is "%s".\n' \
            '    -l WITH_LABEL(0 or 1)\n' \
            '      Whether SAMPLE_FILE contains labels.\n' \
            '      Default is "%d".\n' \
            '    -t THRESHOLD\n' \
            '      Keep features whose frequency are larger than this threshold.\n' \
            '      Default is "%d".\n' % \
            (sys.argv[0], feature_map_filename, with_label, threshold)
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
        elif sys.argv[i] == '-t':
            if i + 1 == len(sys.argv):
                print >> sys.stderr, '"%s" wants a value' % sys.argv[i]
                usage()
            threshold = int(sys.argv[i + 1])
            del sys.argv[i:i + 2]
        else:
            i += 1
        if i == len(sys.argv):
            break

    if len(sys.argv) == 1:
        usage()

    sample_filename_list = sys.argv[1:]
    main(sample_filename_list, with_label)
    save_feature_map(feature_map_filename, threshold)
