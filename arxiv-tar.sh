#!/bin/bash

mkdir -p paper/anc
cp paper/supplement.pdf paper/anc
tar --exclude="paper/*.eps" --exclude="paper/.gitignore" --exclude="*.sh" --exclude="paper/*.out" --exclude="paper/supplement*" --exclude="paper/paper.pdf" -czvf paper.tgz paper/
rm -rf paper/anc
