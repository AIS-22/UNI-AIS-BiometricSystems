#!/bin/bash
codecs=( AVIF BPG HEIC JPEG JPEG2000 JPEG_XL JPEG_XR_0 JPEG_1 JPEG_XR_1 JPEG_XR_2 WEBP )
echo "PLUS"
cd PLUS
find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-25.25s : " "$dir"
  find "$dir" -type f | wc -l
done
#echo "SCUT"
#cd ../SCUT
#find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
#  printf "%-25.25s : " "$dir"
#  find "$dir" -type f | wc -l
#done
#echo "PROTECT"
#cd ../PROTECT
#find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
#  printf "%-25.25s : " "$dir"
#  find "$dir" -type f | wc -l
#done
#echo "IDIAP"
#cd ../IDIAP
#find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
#  printf "%-25.25s : " "$dir"
#  find "$dir" -type f | wc -l
#done
#
#