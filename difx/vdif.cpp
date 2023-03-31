#include <iostream>

#include "lwa.hpp"
#include "vdif.hpp"

VDIFFrame::VDIFFrame(uint16_t id,
                     uint32_t frame_size,
                     uint32_t frames_per_second,
                     uint8_t data_bits=4,
                     bool is_complex=false): _timetag(0), _payload(NULL), _head(0) {
  _frame_size = frame_size;
  _frames_per_second = frames_per_second;
  
  _hdr.station_id = id / 10;
  _hdr.thread_id = id % 10;
  _hdr.is_legacy = 1;
  _hdr.is_invalid = 0;
  _hdr.version = 1;
  _hdr.frame_length = (sizeof(_hdr)+ frame_size) / 8;
  _hdr.log2_nchan = 0;
  _hdr.bits_per_sample_minus_one = data_bits - 1;
  _hdr.is_complex = is_complex;
  
  _payload = (uint8_t *) calloc(_frame_size+4096, 1);
}

void VDIFFrame::set_timetag(uint64_t timetag) {
  _timetag = timetag;
  
  _hdr.seconds_from_epoch = timetag / LWA_FS - 946684800;
  _hdr.frame_in_second = (uint32_t) round((timetag % LWA_FS) / (double) LWA_FS * _frames_per_second);
  if( _hdr.frame_in_second >= _frames_per_second ) {
   _hdr.seconds_from_epoch += 1;
   _hdr.frame_in_second -= _frames_per_second;
  }
}

void VDIFFrame::add_data(uint8_t* data, uint32_t n) {
  ::memcpy(_payload+_head, data, n);
  _head += n;
}

void VDIFFrame::write(std::ofstream &oh) {
  oh.write(reinterpret_cast<char*>(&_hdr), 16);
  oh.write(reinterpret_cast<char*>(_payload), _frame_size);
  
  ::memcpy(_payload, _payload+_frame_size, 4096);
  _head -= _frame_size;
}
