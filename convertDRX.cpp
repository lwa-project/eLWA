#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <set>
#include <cmath>
#include <list>

#include "lwa.hpp"
#include "drx.hpp"
#include "vdif.hpp"


int main(int argc, char** argv) {
  if( argc < 2 ) {
    std::cout << "Must supply a filename to read from" << std::endl;
    return 1;
  } else {
    std::cout << "Reading from '" << argv[1] << "'" << std::endl;
  }
  
  std::string filename = std::string(argv[1]);
  DRXBuffer *buffer = new DRXBuffer(filename);
  int beam = buffer->beam();
  int sample_rate = buffer->sample_rate();
  std::cout << "Start found with sample rate " << sample_rate << std::endl;
  
  int vdif_bits = 4;
  bool vdif_complex = true;
  int vdif_frame_size = 7840;
  double vdif_frame_ns = round(vdif_frame_size * (1e9 / sample_rate) * 10000) / 10000;
  while(    (sample_rate % vdif_frame_size != 0) \
         || ((int) vdif_frame_ns != vdif_frame_ns) \
         || (vdif_frame_size % 8 != 0) ) {
    vdif_frame_size++;
    vdif_frame_ns = round(vdif_frame_size * (1e9 / sample_rate) * 10000) / 10000;
  }
  int vdif_frames_per_second = sample_rate / vdif_frame_size;
  std::cout << "VDIF bits: " << vdif_bits << std::endl;
  std::cout << "Samples per frame: " << vdif_frame_size << " (" << vdif_frame_ns << " ns)" << std::endl;
  std::cout << "Frames per second: " << vdif_frames_per_second << std::endl;
  
  std::string outname1, outname2;
  std::size_t marker = filename.rfind("/");
  outname1 = filename.substr(marker+1, filename.size()-marker);
  marker = outname1.rfind(".");
  outname1 = outname1.substr(0, marker);
  outname2 = outname1+"_b"+std::to_string(beam)+"t2.vdif";
  outname1 = outname1+"_b"+std::to_string(beam)+"t1.vdif";
  
  std::ofstream oh1, oh2;
  oh1.open(outname1, std::ios::out|std::ios::binary);
  oh2.open(outname2, std::ios::out|std::ios::binary);
  
  int s, max_s;
  s = 0;
  max_s = buffer->nframes() / buffer->beampols();
  std::list<DRXFrame> frames;
  std::list<DRXFrame>::iterator frame_it;
  
  uint8_t *mapper;
  mapper = (uint8_t *) calloc(256, 1);
  for(uint16_t i=0; i<256; i++) {
    uint8_t f = (i >> 4) & 0x0F;
    uint8_t s = i & 0x0F;
    f = f < 8 ? f + 8 : f - 8;
    s = s < 8 ? s + 8 : s - 8;
    *(mapper + i) = ((s << 4) & 0xF0) | (f & 0x0F);
  }
  
  VDIFFrame *vdif1X = new VDIFFrame(buffer->beam()*100 + 0, vdif_frame_size, vdif_frames_per_second, 4, true);
  VDIFFrame *vdif1Y = new VDIFFrame(buffer->beam()*100 + 1, vdif_frame_size, vdif_frames_per_second, 4, true);
  VDIFFrame *vdif2X = new VDIFFrame(buffer->beam()*100 + 10, vdif_frame_size, vdif_frames_per_second, 4, true);
  VDIFFrame *vdif2Y = new VDIFFrame(buffer->beam()*100 + 11, vdif_frame_size, vdif_frames_per_second, 4, true);
  
  uint64_t vdif_timetag;
  uint64_t vdif_timetag_step = vdif_frame_size * (buffer->timetag_skip() / 4096);
  
  bool first = true;
  frames = buffer->get();
  while( frames.size() > 0 ) {
    frame_it = std::begin(frames);
    
    DRXFrame frame1X = *(frame_it++);
    DRXFrame frame2X = *(frame_it++);
    DRXFrame frame1Y = *(frame_it++);
    DRXFrame frame2Y = *(frame_it++);
    
    for(int i=0; i<4096; i++) {
      frame1X.payload.bytes[i] = mapper[frame1X.payload.bytes[i]];
      frame1Y.payload.bytes[i] = mapper[frame1Y.payload.bytes[i]];
      frame2X.payload.bytes[i] = mapper[frame2X.payload.bytes[i]];
      frame2Y.payload.bytes[i] = mapper[frame2Y.payload.bytes[i]];
    }
    
    if( first ) {
      vdif_timetag = __bswap_64(frame1X.payload.timetag);
      if( __bswap_16(frame1X.header.time_offset) ) {
        vdif_timetag -= 6600;
      }
      
      if(    ((vdif_timetag % LWA_FS) > (LWA_FS - buffer->timetag_skip()))
          || ((vdif_timetag % LWA_FS) == 0) ) {
        uint32_t sample_offset = (LWA_FS - vdif_timetag % LWA_FS) % LWA_FS;
        sample_offset = sample_offset / (LWA_FS / (uint64_t) buffer->sample_rate());
        
        uint64_t sample_timetag = vdif_timetag + sample_offset*(LWA_FS / (uint64_t) buffer->sample_rate());
        vdif1X->set_timetag(sample_timetag);
        vdif1Y->set_timetag(sample_timetag);
        vdif2X->set_timetag(sample_timetag);
        vdif2Y->set_timetag(sample_timetag);
        
        vdif1X->add_data(&frame1X.payload.bytes[sample_offset], 4096-sample_offset);
        vdif1Y->add_data(&frame1Y.payload.bytes[sample_offset], 4096-sample_offset);
        vdif2X->add_data(&frame2X.payload.bytes[sample_offset], 4096-sample_offset);
        vdif2Y->add_data(&frame2Y.payload.bytes[sample_offset], 4096-sample_offset);
        
        first = false;
      }
      
      frames = buffer->get();
      
      continue;
    }
    
    vdif1X->add_data(&frame1X.payload.bytes[0], 4096);
    vdif1Y->add_data(&frame1Y.payload.bytes[0], 4096);
    vdif2X->add_data(&frame2X.payload.bytes[0], 4096);
    vdif2Y->add_data(&frame2Y.payload.bytes[0], 4096);
    
    if( vdif1X->is_ready() ) {
      vdif1X->write(oh1);
      vdif_timetag = vdif1X->get_timetag() + vdif_timetag_step;
      
      vdif1X->set_timetag(vdif_timetag);
    }
    if( vdif1Y->is_ready() ) {
      vdif1Y->write(oh1);
      
      vdif_timetag = vdif1Y->get_timetag() + vdif_timetag_step;
      vdif1Y->set_timetag(vdif_timetag);
    }
    
    if( vdif2X->is_ready() ) {
      vdif2X->write(oh2);
      
      vdif_timetag = vdif2X->get_timetag() + vdif_timetag_step;
      vdif2X->set_timetag(vdif_timetag);
    }
    if( vdif2Y->is_ready() ) {
      vdif2Y->write(oh2);
      
      vdif_timetag = vdif2Y->get_timetag() + vdif_timetag_step;
      vdif2Y->set_timetag(vdif_timetag);
    }
    
    frames = buffer->get();
    
    s += 1;
    if( s % 5000 == 0 ) {
      std::cout << "\r" << "At " << (int) 100.0*s/max_s << "%" << std::flush;
    }
  }
  std::cout << "\r" << "At " << (int) 100.0*s/max_s << "%" << std::endl;
  
  ::free(mapper);
  
  oh1.close();
  oh2.close();
  
  return 0;
}
