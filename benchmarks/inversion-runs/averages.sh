#!/bin/bash

cd invocation-dir

latex=""
for i in 3 10 17; do
    
    title=$((i - 2))
    name=$(sed "${title}q;d" inv-run-1.txt)
    
    numbersonly=$(echo $name | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g' | tr -s ' ' | sed 's/ /\n/g')

    tail -n +$i inv-run-1.txt | head -n 4 | tr -s " " | cut -d" " -f8 > "/tmp/cols"
    kerntype=$(sed "2q;d" inv-run-1.txt | cut -d" " -f12)

    echo $name

    t=$(mktemp)

    for file in inv-run-*.txt; do
        tail -n +$i $file | head -n 4 | tr -s " " | cut -d" " -f4 > "${t}-column-$file"
    done

    paste $t-column-* > $t
    awk '{s=0; for (i=1;i<=NF;i++)s+=$i; print s/NF;}' $t > "/tmp/avgs"
    paste /tmp/avgs /tmp/cols

    minnum=$(sort -n /tmp/avgs | head -n 1)

    row=""
    for number in $numbersonly "\\\\texttt{$kerntype}"  "0.1" $(cat /tmp/avgs); do
        if [ "$number" = "$minnum" ]; then
            left="\\\\textbf{"
            right="}"
        else
            left=""
            right=""
        fi
        row="${row} & \$ ${left}${number}${right} \$"
    done
    row=$(echo $row | cut -c3-)
    newline=$'\n'
    latex="$latex $row \\\\\\\\ $newline"
done

echo
echo 'in latex'

printf "${latex}"
