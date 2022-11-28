#ifndef __INCLUDE_DRX_HPP_
#define __INCLUDE_DRX_HPP_

#include <string>
#include <fstream>
#include <stdexcept>
#include <map>
#include <set>
#include <list>

#if defined(__linux__)
/* Linux */
#include <byteswap.h>
#elif defined(__APPLE__) && defined(__MACH__)
/* OSX */
#include <libkern/OSByteOrder.h>
#define __bswap_16 OSSwapInt16
#define __bswap_32 OSSwapInt32
#define __bswap_64 OSSwapInt64
#endif

#include "lwa.hpp"

typedef struct __attribute__((packed)) {
    uint32_t sync_word;
    union {
        struct {
            uint32_t frame_count:24;
            uint8_t id:8;
        };
        /* Also: 
            struct {
                uint32_t frame_count:24;
                uint8_t  beam:3;
                uint8_t  tune:3;
                uint8_t  reserved:1;
                uint8_t  pol:1;
            };
        */
        uint32_t frame_count_word;
    };
    uint32_t second_count;
    uint16_t decimation;
    uint16_t time_offset;
} DRXHeader;


typedef struct __attribute__((packed)) {
    uint64_t timetag;
    uint32_t tuning_word;
    uint32_t flags;
    uint8_t  bytes[4096];
} DRXPayload;


typedef struct __attribute__((packed)) {
    DRXHeader header;
    DRXPayload payload;
} DRXFrame;

#define DRX_GET_BEAM(x) (((x.header.frame_count_word & 0xFF) >> 0) & 0x07)
#define DRX_GET_TUNE(x) (((x.header.frame_count_word & 0xFF) >> 3) & 0x07)
#define DRX_GET_POLN(x) (((x.header.frame_count_word & 0xFF) >> 7) & 0x01)

class DRXBuffer {
private:
  std::string _filename;
  std::ifstream _fh;
  
  int _nframes;
  int _sample_rate;
  int _timetag_skip;
  LWATime _start;
  uint64_t _last_timetag;
  
  std::set<uint32_t> _frame_ids;
  std::set<uint32_t>::iterator _id_it;
  
  std::map<uint32_t, DRXFrame>::iterator _set_it;
  std::map<uint64_t, std::map<uint32_t, DRXFrame> > _buffer;
  std::map<uint64_t, std::map<uint32_t, DRXFrame> >::iterator _buff_it;
  
public:
  DRXBuffer(std::string filename);
  ~DRXBuffer() {
    _fh.close();
  }
  inline int beam() { 
    _id_it = std::begin(_frame_ids);
    return (*_id_it & 0x07);
  }
  inline int beampols()     { return _frame_ids.size(); }
  inline int sample_rate()  { return _sample_rate;  }
  inline int timetag_skip() { return _timetag_skip; }
  inline int nframes()      { return _nframes;      }
  inline LWATime start()    { return _start;        }
  double offset(double step);
  std::list<DRXFrame> get();
  inline void reset() {
    _last_timetag = 0;
    _buffer.clear();
  }
};

#endif // __INCLUDE_DRX_HPP_
