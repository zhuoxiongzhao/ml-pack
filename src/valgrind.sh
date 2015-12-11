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
output="valgrind.out"

set -e
make
rm -f $output

function run() {
    cmd=$*
    echo "running $cmd" >> $output
    $valgrind $cmd >> $output
}

run ./lr-test

run ./lr-main train test-data/heart_scale.unordered
cat model >> $output
run ./lr-main train test-data/heart_scale -o model
cat model >> $output

run ./lr-main predict test-data/heart_scale -l 1 -o pred
cat pred >> $output
run ./lr-main predict test-data/heart_scale.bad -l 1 -o pred
cat pred >> $output
run ./lr-main predict test-data/heart_scale.nolabel -l 0 -o pred
cat pred >> $output
run ./lr-main predict test-data/heart_scale.nolabel.bad -l 0 -o pred
cat pred >> $output

run ./lr-main train test-data/heart_scale -o model -ft 1 -d 1024
cat model >> $output
run ./lr-main predict -m model test-data/heart_scale.nolabel -l 0 -o pred -ft 1 -d 1024
cat pred >> $output
run ./lr-main train test-data/heart_scale -o model -ft 1 -d 1024 -b 0
cat model >> $output
run ./lr-main predict -m model test-data/heart_scale.nolabel -l 0 -o pred -ft 1 -d 1024 -b 0
cat pred >> $output

run ./lr-main.exe train test-data/heart_scale -t 0.3 >> $output
cat model >> $output

run ./lr-main.exe train test-data/heart_scale -cv 3 >> $output
cat model0 >> $output
cat model1 >> $output
cat model2 >> $output

run ./gen-feature-map test-data/samples2 -l 1 -o test-data/samples2.feature-map
run ./map-sample -f test-data/samples2.feature-map -l 1 test-data/samples2 -o test-data/samples2.libsvm

run ./gen-feature-map test-data/samples3 -l 0 -o test-data/samples3.feature-map
run ./map-sample -f test-data/samples3.feature-map -l 0 test-data/samples3 -o test-data/samples3.libsvm

run ./problem-gen-bin test-data/heart_scale
run ./problem-load-bin test-data/heart_scale.bin

rm -f model* pred
rm -f test-data/heart_scale.bin
