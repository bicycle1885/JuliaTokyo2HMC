#!/usr/bin/env bash

convert $1 +flatten -alpha on -channel A -evaluate set 30% +channel $2
