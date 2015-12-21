#! /bin/bash

DL_CMD="curl -C - -O"

if [ ! -f "20news-18828.tar.gz" ]; then
    $DL_CMD http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz
fi
if [ ! -d "20news-18828" ]; then
    tar zxvf 20news-18828.tar.gz
fi
