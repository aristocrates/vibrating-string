#!/bin/bash

cd $1
mkdir workspace
cp *.png workspace
cd workspace
mogrify -resize 480x360 *.png
convert -delay 1 -loop 0 *.png $1.gif
cp $1.gif ../../animated_gif/$1.gif
cd ../..
