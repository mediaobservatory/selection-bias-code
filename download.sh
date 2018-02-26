#!/bin/bash

cat masterfilelist.txt | cut -d ' ' -f 3 | grep mentions | xargs wget -P dl/
