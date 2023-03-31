#ifndef __INCLUDE_VDIF_HPP_
#define __INCLUDE_VDIF_HPP_

#include <fstream>
#include <cstring>

typedef struct __attribute__((packed)) {
    struct {
        uint32_t seconds_from_epoch:30;
        uint8_t  is_legacy:1;
        uint8_t  is_invalid:1;
    };
    struct {
        uint32_t frame_in_second:24;
        uint16_t refEpoch:6;
        uint8_t  unassigned:2;
    };
    struct {
        uint32_t frame_length:24;
        uint32_t log2_nchan:5;
        uint8_t  version:3;
    };
    struct {
        uint16_t station_id:16;
        uint16_t thread_id:10;
        uint8_t  bits_per_sample_minus_one:5;
        uint8_t  is_complex:1;
    };
} VDIFBasicHeader;

typedef struct __attribute__((packed)) {
    uint32_t extended_data_1;
    uint32_t extended_data_2;
    uint32_t extended_data_3;
    uint32_t extended_data_4;
} VDIFExtendedHeader;

class VDIFFrame {
private:
  uint32_t        _frame_size;
  uint32_t        _frames_per_second;
  uint64_t        _timetag;
  VDIFBasicHeader _hdr;
  uint8_t*        _payload;
  uint32_t        _head;
  
public:
  VDIFFrame(uint16_t id, uint32_t frame_size, uint32_t frames_per_second, uint8_t data_bits, bool is_complex);
  ~VDIFFrame() {
    if( _payload != NULL ) {
      ::free(_payload);
    }
  }
  void set_timetag(uint64_t timetag);
  inline uint64_t get_timetag() { return _timetag; }
  void add_data(uint8_t* data, uint32_t n);
  inline bool is_ready() { return (_head >= _frame_size); }
  void write(std::ofstream &oh);
};

#endif // __INCLUDE_VDIF_HPP_
