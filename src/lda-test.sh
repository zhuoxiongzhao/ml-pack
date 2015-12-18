#! /bin/bash
#
# Author: Yafei Zhang (zhangyafeikimi@gmail.com)
#

valgrind=$(which valgrind 2>/dev/null)
if test x$valgrind != "x"; then
    valgrind="$valgrind --tool=memcheck"
fi
output="lda-test.out"
data_dir="lda-test-data"

set -e
make
rm -f $output

function run() {
    cmd=$*
    echo "running $cmd" >> $output
    $valgrind $cmd >> $output
}

run ./lda-test
rm -f yahoo-*

