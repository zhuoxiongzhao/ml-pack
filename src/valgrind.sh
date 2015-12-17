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
data_dir="lr-test-data"

set -e
make
rm -f $output

function run() {
    cmd=$*
    echo "running $cmd" >> $output
    $valgrind $cmd >> $output
}

run ./lr-test

run ./lr-main train $data_dir/heart_scale.unordered
cat model >> $output
run ./lr-main train $data_dir/heart_scale -o model
cat model >> $output

run ./lr-main predict $data_dir/heart_scale -l 1 -o pred
cat pred >> $output
run ./lr-main predict $data_dir/heart_scale.bad -l 1 -o pred
cat pred >> $output
run ./lr-main predict $data_dir/heart_scale.nolabel -l 0 -o pred
cat pred >> $output
run ./lr-main predict $data_dir/heart_scale.nolabel.bad -l 0 -o pred
cat pred >> $output

run ./lr-main train $data_dir/heart_scale -o model -ft 1 -d 1024
cat model >> $output
run ./lr-main predict -m model $data_dir/heart_scale.nolabel -l 0 -o pred -ft 1 -d 1024
cat pred >> $output
run ./lr-main train $data_dir/heart_scale -o model -ft 1 -d 1024 -b 0
cat model >> $output
run ./lr-main predict -m model $data_dir/heart_scale.nolabel -l 0 -o pred -ft 1 -d 1024 -b 0
cat pred >> $output

run ./lr-main.exe train $data_dir/heart_scale -t 0.3 >> $output
cat model >> $output

run ./lr-main.exe train $data_dir/heart_scale -cv 3 >> $output
cat model0 >> $output
cat model1 >> $output
cat model2 >> $output

run ./gen-feature-map $data_dir/samples2 -l 1 -o $data_dir/samples2.feature-map
run ./map-sample -f $data_dir/samples2.feature-map -l 1 $data_dir/samples2 -o $data_dir/samples2.libsvm

run ./gen-feature-map $data_dir/samples3 -l 0 -o $data_dir/samples3.feature-map
run ./map-sample -f $data_dir/samples3.feature-map -l 0 $data_dir/samples3 -o $data_dir/samples3.libsvm

run ./problem-gen-bin $data_dir/heart_scale
run ./problem-load-bin $data_dir/heart_scale.bin

rm -f model* pred
rm -f $data_dir/heart_scale.bin
