#!/bin/bash

function epsconvert {
    mv $1.eps $1-unbounded.eps
    epstool --quiet --copy --bbox $1-unbounded.eps $1.eps
    epstopdf --quiet --hires $1.eps
    rm $1-unbounded.eps
}

echo 'iterations.pdf'
cp ../benchmarks/pictures/out/iterations.pdf .

echo 'running_cutoff.pdf'
cp ../benchmarks/pictures/out/running_cutoff.pdf .

echo 'relgrad_l2.eps -> relgrad_l2.pdf'
cp ../benchmarks/grad-grid/out/relgrad_l2.eps .
epsconvert relgrad_l2

echo 'time_ratio.eps -> time_ratio.pdf'
cp ../benchmarks/grad-grid/out/time_ratio.eps .
epsconvert time_ratio

echo 'relalpha_l2.eps -> relalpha_l2.pdf'
cp ../benchmarks/grad-grid/out/relalpha_l2.eps .
epsconvert relalpha_l2

echo 'fx2007graph.pdf'
cp ../benchmarks/pictures/out/fx2007graph.pdf .

echo 'weather.pdf'
cp ../benchmarks/pictures/out/weather.pdf .

echo 'representation-cmp/out/results.tex -> representation.tex'
cp ../benchmarks/representation-cmp/out/results.tex representation.tex

echo 'results_fx2007.tex'
cp ../benchmarks/fx2007-out/results_fx2007.tex .

echo 'results_weather.tex'
cp ../benchmarks/weather-out/results_weather.tex .

echo 'results_synth.tex'
cp ../benchmarks/synth/out/results_synth.tex .
