#!/bin/bash

echo 'iterations.eps'
cp ../benchmarks/picture-fx2007/out/iterations.eps .

echo 'running_cutoff.eps'
cp ../benchmarks/picture-fx2007/out/running_cutoff.eps .

echo 'relgrad_l2.eps'
cp ../benchmarks/grad-grid/out/relgrad_l2.eps .

echo 'time_ratio.eps'
cp ../benchmarks/grad-grid/out/time_ratio.eps .

echo 'relalpha_l2.eps'
cp ../benchmarks/grad-grid/out/relalpha_l2.eps .

echo 'fx2007graph.pdf'
cp ../benchmarks/picture-fx2007/out/fx2007graph.pdf .

echo 'm_time_nlpd.eps'
cp ../benchmarks/weather/out/m_time_nlpd.eps .

echo 'representation-cmp/out/results.tex -> representation.tex'
cp ../benchmarks/representation-cmp/out/results.tex representation.tex

echo 'results_fx2007.tex'
cp ../benchmarks/fx2007/out/results_fx2007.tex .

echo 'results_weather.tex'
cp ../benchmarks/weather/out/results_weather.tex .
