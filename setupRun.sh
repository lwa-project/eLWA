#!/bin/bash

#
# setupRun.sh - Simple script to setup a directory to correlate an observing run
#

# Argument parsing
while [[ $# -gt 0 ]]; do
	key="${1}"
	case ${key} in
		-h|--help)
			echo "setupRun.sh - Script to help get ready for a correlation run"
			echo " "
			echo "Usage:"
			echo "setupRun.sh [OPTIONS] directory"
			echo " "
			echo "Options:"
			echo "-h, --help     Show this help message"
			exit 0
			;;
		*)
			;;
	esac
	shift
done

# Setup
## Directory to create/populate
DIR=${key}
## Base directory where all of the script reside
BASE=`readlink -f ${0} | xargs dirname `
## Script that need to be linked to
FILES=( multirate.py  utils.py \
creatConfigFile.py superCorrelator.py plotConfig.py plotUVCoverage.py plotFringes2.py \
fringeSearchZero.py fringeSearchFine.py \
buildIDI.py )

# Create the directory, if needed
if [[ ! -x ${DIR} ]]; then
	echo "Creating directory '${DIR}'"
	mkdir -p ${DIR}
else
	echo "Directory '${DIR}' already exists, skipping creation"
fi

# Populate the directory with symbolic links
for FILE in ${FILES[@]}; do
	echo "  Linking '${FILE}"
	ln -sf ${BASE}/${FILE} ${DIR}/${FILE}
done

# Done
echo "Done"
