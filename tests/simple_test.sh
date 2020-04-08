#!/bin/bash

# Setup
## Exit when any command fails
set -e
## Keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
## echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# Download the raw data
curl https://fornax.phys.unm.edu/lwa/data/eLWA_test_raw.tar.gz -o eLWA_test_raw.tar.gz
tar xzf eLWA_test_raw.tar.gz

# Write a stub get_vla_ant_pos.py to use if we need it
if [[ ! -e ../get_vla_ant_pos.py ]]; then
    echo -e "class database(object):\n\
    def __init__(self, *args, **kwds):\n\
        self._ready = True\n\
    def get_pad(self,ant,date):\n\
        return 'W40', None\n\
    def get_xyz(self,ant,date):\n\
        return (-6777.061, -360.702, -3550.947)\n\
    def close(self):\n\
        return True\n" > ../get_vla_ant_pos.py
fi

# Correlate eLWA data, build a FITS-IDI file, and flag it
python ../createConfigFile.py -o elwa.config *
python ../superCorrelator.py -t 1 -l 256 -j -g elwa elwa.config
python ../buildIDI.py -t elwa elwa-*.npz
python ../flagDelaySteps.py buildIDI_elwa.FITS_1
python ../flagIDI.py buildIDI_elwa_flagged.FITS_1

# Correlate LWA-only data, build a FITS-IDI file, and flag it
python ../createConfigFile.py -o lwaonly.config 0* *.tgz
python ../superCorrelator.py -t 1 -l 256 -j -w 1 -g lwaonlyL lwaonly.config
python ../superCorrelator.py -t 1 -l 256 -j -w 2 -g lwaonlyH lwaonly.config
python ../buildMultiBandIDI.py -t lwaonly lwaonly[LH]-*.npz
python ../flagDelaySteps.py buildIDI_lwaonly.FITS_1
python ../flagIDI.py buildIDI_lwaonly_flagged.FITS_1

# Download the reference data
mkdir -p ref && cd ref
curl https://fornax.phys.unm.edu/lwa/data/eLWA_test_ref.tar.gz -o eLWA_test_ref.tar.gz
tar xzf eLWA_test_ref.tar.gz
cd ..

# Compare
python ./compare_results.py buildIDI_elwa_flagged_flagged.FITS_1 ref/buildIDI_elwa_flagged_flagged.FITS_1
python ./compare_results.py buildIDI_lwaonly_flagged_flagged.FITS_1 ref/buildIDI_lwaonly_flagged_flagged.FITS_1
