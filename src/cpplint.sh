#! /bin/bash
#
# Author: Yafei Zhang (zhangyafeikimi@gmail.com)
#
# check code style
#

python cpplint.py $(find . -name "*.cc" -or -name "*.h" | grep -v "city\.cc\|city\.h\|mt64\.h\|mt19937ar\.h" | xargs)
