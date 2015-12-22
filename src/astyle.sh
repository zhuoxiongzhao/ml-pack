#! /bin/bash
#
# Author: Yafei Zhang (zhangyafeikimi@gmail.com)
#
# format code with astyle
#

astyle --style=google \
--add-brackets \
--align-pointer=type \
--align-reference=type \
--indent-col1-comments \
--indent-preproc-define \
--indent=spaces=2 \
--lineend=linux \
--pad-header \
--pad-oper \
-n \
$(find . -name "*.c" -or -name "*.cc" -or -name "*.h" | xargs)
