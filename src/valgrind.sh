#! /bin/bash
#
# Author: Yafei Zhang (zhangyafeikimi@gmail.com)
#
# do valgrind test
#
valgrind=$(which valgrind 2>/dev/null)
if test x$valgrind != "x"; then
    valgrind="$valgrind --tool=memcheck"
fi

set -e

make
$valgrind ./lr-test

$valgrind ./lr-main train test-data/heart_scale -o model1
$valgrind ./lr-main train test-data/heart_scale.unordered -o model2

$valgrind ./lr-main predict -m model1 test-data/heart_scale -l 1 -o pred
wc -l pred
rm pred
$valgrind ./lr-main predict -m model1 test-data/heart_scale.bad -l 1 -o pred
wc -l pred
rm pred
$valgrind ./lr-main predict -m model1 test-data/heart_scale.nolabel -l 0 -o pred
wc -l pred
rm pred
$valgrind ./lr-main predict -m model1 test-data/heart_scale.nolabel.bad -l 0 -o pred
wc -l pred
rm model1 model2 pred

$valgrind ./lr-main train test-data/heart_scale -o model -ft 1 -d 1024
$valgrind ./lr-main predict -m model test-data/heart_scale.nolabel -l 0 -o pred -ft 1 -d 1024
rm model pred
$valgrind ./lr-main train test-data/heart_scale -o model -ft 1 -d 1024 -b 0
$valgrind ./lr-main predict -m model test-data/heart_scale.nolabel -l 0 -o pred -ft 1 -d 1024 -b 0
rm model pred

$valgrind ./gen-feature-map test-data/samples2 -l 1 -o test-data/samples2.feature-map
$valgrind ./map-sample -f test-data/samples2.feature-map -l 1 test-data/samples2 -o test-data/samples2.libsvm

$valgrind ./gen-feature-map test-data/samples3 -l 0 -o test-data/samples3.feature-map
$valgrind ./map-sample -f test-data/samples3.feature-map -l 0 test-data/samples3 -o test-data/samples3.libsvm

./problem-gen-bin test-data/heart_scale
./problem-load-bin test-data/heart_scale.bin
rm test-data/heart_scale.bin
