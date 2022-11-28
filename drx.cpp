#include <iostream>
#include <cstring>

#include "drx.hpp"
#include "lwa.hpp"

DRXBuffer::DRXBuffer(std::string filename): _sample_rate(0), _timetag_skip(0), _last_timetag(0) {
  _filename = filename;
  _fh.open(filename, std::ios::in|std::ios::binary);
  if( !_fh.good() ) {
    throw(std::runtime_error("Failed to open file"));
  }
  
  // Find valid data
  DRXFrame frame;
  _fh.read(reinterpret_cast<char*>(&frame), sizeof(frame));
  while( _fh.good() && (   (frame.header.sync_word != 0x5CDEC0DE) \
                        || (frame.header.decimation == 0) ) ) {
    _fh.seekg(1-sizeof(frame), std::ios_base::cur);
    _fh.read(reinterpret_cast<char*>(&frame), sizeof(frame));
  }
  if( !_fh.good() ) {
    throw(std::runtime_error("Failed to find valid data"));
  }
  _start = timetag_to_lwatime(__bswap_64(frame.payload.timetag));
  _fh.seekg(-sizeof(frame), std::ios_base::cur);
  
  // Determine the sample rate and timetag skip
  _sample_rate = LWA_FS / __bswap_16(frame.header.decimation);
  _timetag_skip = 4096 * __bswap_16(frame.header.decimation);
  
  // Determine the how many tuning/polarization pairs are in the data
  long marker = _fh.tellg();
  for(int i=0; i<128; i++) {
    _fh.read(reinterpret_cast<char*>(&frame), sizeof(frame));
    if( !_fh.good() ) {
      break;
    }
    _frame_ids.insert(frame.header.frame_count_word);
  }
  _fh.seekg(marker, std::ios::beg);
  
  // Determine how many frames are in the file
  _fh.seekg (0, std::ios::end);
  _nframes = _fh.tellg() / sizeof(frame);
  _fh.seekg(marker, std::ios::beg);
}

double DRXBuffer::offset(double step) {
  LWATime t, t_start, t_offset;
  
  DRXFrame frame;
  _fh.read(reinterpret_cast<char*>(&frame), sizeof(frame));
  _fh.seekg(-sizeof(frame), std::ios_base::cur);
  
  t_start = t = timetag_to_lwatime(__bswap_64(frame.payload.timetag));
  t_offset = lwatime_offset(t_start, step);
  
  int nread = 0;
  while( _fh.good() && (lwatime_diff(t_offset, t) > 0.0) ) {
    _fh.read(reinterpret_cast<char*>(&frame), sizeof(frame));
    t = timetag_to_lwatime(__bswap_64(frame.payload.timetag));
    nread++;
  }
  _fh.seekg(-sizeof(frame), std::ios_base::cur);
  nread--;
  
  _start = t;
  _nframes = _nframes - nread;
  this->reset();
  
  return lwatime_diff(t, t_start);
}

std::list<DRXFrame> DRXBuffer::get() {
  DRXFrame frame;
  uint64_t timetag;
  while(   (_buffer.size() < 20) \
        && (_fh.read(reinterpret_cast<char*>(&frame), sizeof(frame)).good()) ) {
      timetag = __bswap_64(frame.payload.timetag);
      
      // See if we have already dumped this timetag
      if( timetag < std::begin(_buffer)->first ) {
        continue;
      }
      
      // Add it to the buffer
      // If we have a new timetag, create an entry for it
      if( _buffer.count(timetag) == 0 ) {
        std::map<uint32_t, DRXFrame> frame_set;
        _buffer[timetag] = frame_set;
      }
      
      // Save the frame
      _buffer[timetag][frame.header.frame_count_word] = frame;
  }
  
  std::list<DRXFrame> output;
  // Loop over ordered ID numbers
  if( _buffer.size() > 0 ) {
    _buff_it = std::begin(_buffer);
    
    // But first, check for any small gaps in the buffer by looking at the time
    // tag coming out vs. what we previously had
    if( _last_timetag > 0 ) {
      double missing = (_buff_it->first - _last_timetag - _timetag_skip) / _timetag_skip;
      
      // If it looks like we are missing something, fill the gap... up to a point
      if( (missing > 0) && ((int) missing == missing) && (missing < 50) ) {
        DRXFrame dummy_frame;
        ::memcpy(&dummy_frame, &std::begin(_buff_it->second)->second, sizeof(dummy_frame));
        ::memset(&(dummy_frame.payload.bytes), 0, 4096);
        uint64_t dummy_timetag = _buff_it->first;
        
        for(int j=0; j<missing; j++) {
          std::map<uint32_t, DRXFrame> frame_set;
          _buffer[dummy_timetag + j*_timetag_skip] = frame_set;
          
          for(_id_it=std::begin(_frame_ids); _id_it!=std::end(_frame_ids); _id_it++) {
            dummy_frame.header.frame_count_word = *_id_it;
            dummy_frame.payload.timetag = __bswap_64(dummy_timetag + j*_timetag_skip);
            
            _buffer[dummy_timetag + j*_timetag_skip][*_id_it] = dummy_frame;
          }
        }
        
      _buff_it = std::begin(_buffer);
      }
    }
    
    _last_timetag = _buff_it->first;
    for(_id_it=std::begin(_frame_ids); _id_it!=std::end(_frame_ids); _id_it++) {
      _set_it = _buff_it->second.find(*_id_it);
      
      if( _set_it != std::end(_buff_it->second) ) {
        // Great, we have the frame
        frame = _set_it->second;
      } else {
        // Boo, we need to create a fake frame
        ::memcpy(&frame, &std::begin(_buff_it->second)->second, sizeof(frame));
        frame.header.frame_count_word = *_id_it;
        ::memset(&(frame.payload.bytes), 0, 4096);
      }
      output.push_back(frame);
    }
    
    if( output.size() > 0 ) {
      _buffer.erase(_buff_it->first);
    }
  }
  
  return output;  
}
