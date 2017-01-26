#!/bin/bash

set -e

repo="$PWD"

cd /tmp/
if [ -d "runlmc" ]; then
    rm -rf runlmc
fi

git clone -b gh-pages git@github.com:vlad17/runlmc.git

cd runlmc

cp -fr $repo/doc/_generated/_build/. .
cp runlmc.html index.html

git add .
git commit -m "updated docs"
git push origin gh-pages


