#!/bin/bash

tar --exclude="paper/*.eps" --exclude="paper/.gitignore" --exclude="*.sh" --exclude="paper/*.out" --exclude="paper/supplement*" --exclude="paper/paper.pdf" -czvf paper.tgz paper/
