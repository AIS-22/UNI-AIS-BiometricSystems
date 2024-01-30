#!/bin/bash

echo "PLUS"
cd PLUS
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done
echo "SCUT"
cd ../SCUT
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done
echo "PROTECT"
cd ../PROTECT
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done
echo "IDIAP"
cd ../IDIAP
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done

