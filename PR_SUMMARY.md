# Dual-Tuning Support for LWA-Only Observations

## Summary

This PR implements dual-tuning support for `superCorrelator.py` and `superPulsarCorrelator.py`, allowing both DRX tunings to be processed in a single invocation for LWA-only observations (when no VDIF files are present). This eliminates the need for two separate correlator runs.

## Motivation

Previously, processing both DRX tunings required two separate invocations of the correlator with `-w 1` and `-w 2`. For LWA-only observations where VDIF data is not constraining the tuning selection, this was inefficient. The new `-w 0` option processes both tunings simultaneously while maintaining backward compatibility.

## Changes to `superCorrelator.py`

### Data Structure
- Changed `dataD` from 2D `(nPols, samples)` to 3D `(2, nPols, samples)` with tunings-first ordering to maintain C-contiguity for the F-engine
- Updated array initialization and zeroing to work with 3D structure

### Frame Reading
- Removed tuning filter for DRX-only data: `if tune != vdifPivot: continue`
- Fixed frame reading loop to read 2x frames: `while k < beampols[j]*nFramesD*2`
- Properly index into 3D array: `dataD_view[tid, bid, ...]` where `tid = tune - 1`

### Processing Logic
- Added tuning selection based on VDIF presence:
  - VDIF + DRX: Process only VDIF-matching tuning
  - VDIF only: Process using dummy tuning value
  - DRX only with `-w 0`: Process both tunings
  - DRX only with `-w 1/2`: Process specified tuning
- Moved VDIF f-engine outside tuning loop (runs once per sub-integration)
- Implemented per-tuning state tracking with dictionaries

### Output Files
- Dual-tuning mode: `{base}L-vis2-{count}.npz` and `{base}H-vis2-{count}.npz`
- Single-tuning mode: `{base}-vis2-{count}.npz` (unchanged)
- Compatible with existing `buildMultiBandIDI.py` expectations

## Changes to `superPulsarCorrelator.py`

Applied same dual-tuning architecture with additional pulsar-specific features:

### Pulsar Binning Support
- Per-tuning bin arrays for phase folding across `nProfileBins`
- Per-tuning state tracking including frequency arrays, visibility accumulators, and antenna indices
- Output files: `{base}[LH]-vis2-bin{bin:03d}-{count:05d}.npz` for dual-tuning

### Bug Fixes in `superPulsarCorrelator.py`
1. **Line 124**: Fixed reader type checking from `elif readers[i] is (drx, drx8):` to `else:`
   - The `is` operator doesn't work for tuple membership checking
   - Caused `IndexError` when accessing `beampols[i]`

2. **Frequency array storage**: Added per-tuning frequency arrays
   - Both L and H files were saving the same frequency array
   - Now stores `tuning_state[tid]['freqXX']` per tuning

## Changes to `flagIDI.py`

Fixed crash when all scans are skipped due to insufficient integrations:
- Check if `nFlags == 0` before building FLAG table
- Skip FLAG table creation with warning instead of crashing with `IndexError`
- Properly handle edge case of pulsar data with few integrations per bin

## Test Suite Additions

### `tests/test_lwaonly.py`
- `test_1a_correlate_dual_tuning`: Run correlator with `-w 0` on both tunings
- `test_1b_compare_dual_vs_single`: Validate dual-tuning output matches single-tuning

### `tests/test_pulsar.py` (New)
Complete test suite for pulsar correlator:
- Synthetic polyco file generation for 1-second period "pulsar"
- Manual config file creation (LWA1 + LWA-SV, 2 antennas)
- Single-tuning correlation test
- Dual-tuning correlation test with frequency validation
- FITS-IDI building from binned data
- RFI flagging
- Fringe searching

Uses test data from existing LWA-only dataset with fake pulsar binning applied to continuum data.

## Bugs Fixed During Development

1. **Insufficient frame reading**: Loop condition wasn't multiplied by 2 after removing tuning filter
2. **VDIF processing placement**: Moved outside tuning loop to prevent skipped sub-integrations
3. **Output file naming**: Changed from `-t1-`/`-t2-` to `L-`/`H-` for `buildMultiBandIDI.py` compatibility
4. **Array zeroing syntax**: Updated for 3D structure using `dataD_view[...] = 0`
5. **VDIF-only edge case**: Use dummy tuning value `[0]` instead of empty list
6. **Polyco format**: Fixed pulsar name, MJD, and config file format for test suite
7. **Polarization format**: Changed from `Pols XX,YY` to `Pols X, Y` in test config

## Performance Impact

- **Memory**: Doubles memory for DRX data buffer (stores both tunings)
- **Computation**: Doubles F-engine operations when processing both tunings
- **Benefit**: Eliminates need for second correlator invocation, saving I/O and setup overhead

## Backward Compatibility

- Existing behavior unchanged for VDIF+DRX observations
- Existing `-w 1` and `-w 2` options work as before
- New `-w 0` option enables dual-tuning mode
- Output file formats unchanged except for L/H suffix in dual-tuning mode

## Testing

All existing tests pass, plus new tests added:
- ✅ eLWA tests (VDIF + DRX)
- ✅ LWA-only tests (single and dual tuning)
- ✅ Pulsar correlator tests (single and dual tuning, with fringe search)

## Related Issues

Implements dual-tuning processing for LWA-only observations to improve efficiency by eliminating redundant correlator invocations.
