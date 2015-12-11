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

rm -f valgrind.out
$valgrind ./lr-main train test-data/heart_scale -o model1
cat model1 >> valgrind.out
$valgrind ./lr-main train test-data/heart_scale.unordered -o model2
cat model2 >> valgrind.out

$valgrind ./lr-main predict -m model1 test-data/heart_scale -l 1 -o pred
cat pred >> valgrind.out
$valgrind ./lr-main predict -m model1 test-data/heart_scale.bad -l 1 -o pred
cat pred >> valgrind.out
$valgrind ./lr-main predict -m model1 test-data/heart_scale.nolabel -l 0 -o pred
cat pred >> valgrind.out
$valgrind ./lr-main predict -m model1 test-data/heart_scale.nolabel.bad -l 0 -o pred
cat pred >> valgrind.out

$valgrind ./lr-main train test-data/heart_scale -o model -ft 1 -d 1024
cat model >> valgrind.out
$valgrind ./lr-main predict -m model test-data/heart_scale.nolabel -l 0 -o pred -ft 1 -d 1024
cat pred >> valgrind.out
$valgrind ./lr-main train test-data/heart_scale -o model -ft 1 -d 1024 -b 0
cat model >> valgrind.out
$valgrind ./lr-main predict -m model test-data/heart_scale.nolabel -l 0 -o pred -ft 1 -d 1024 -b 0
cat pred >> valgrind.out

$valgrind ./gen-feature-map test-data/samples2 -l 1 -o test-data/samples2.feature-map
$valgrind ./map-sample -f test-data/samples2.feature-map -l 1 test-data/samples2 -o test-data/samples2.libsvm

$valgrind ./gen-feature-map test-data/samples3 -l 0 -o test-data/samples3.feature-map
$valgrind ./map-sample -f test-data/samples3.feature-map -l 0 test-data/samples3 -o test-data/samples3.libsvm

$valgrind ./problem-gen-bin test-data/heart_scale
$valgrind ./problem-load-bin test-data/heart_scale.bin

rm model model1 model2 pred
rm test-data/heart_scale.bin
