#!/bin/bash

make -j 24 -C .. duals-mov.exe

../duals-mov.exe duals.mov
mv duals.mov /d/video/ruperts/
