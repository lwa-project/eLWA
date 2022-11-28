#ifndef __INCLUDE_LWA_HPP_
#define __INCLUDE_LWA_HPP_

#include <utility>
#include <cmath>
#include <ctime>
#include <string>

#define LWA_FS 196000000

/*
 Time conversion functions
*/

typedef std::pair<uint64_t,double> LWATime;

inline LWATime timetag_to_lwatime(uint64_t timetag) {
  LWATime lwatime;
  lwatime.first = timetag / LWA_FS;
  lwatime.second = (timetag - lwatime.first*LWA_FS) / (double) LWA_FS;
  return lwatime;
}

inline uint64_t lwatime_to_timetag(LWATime lwatime) {
  uint64_t timetag = lwatime.first * LWA_FS;
  timetag += (uint64_t) round(lwatime.second * LWA_FS);
  return timetag;
}

inline LWATime lwatime_offset(LWATime lwatime, double offset) {
  LWATime new_lwatime;
  new_lwatime.first = lwatime.first + (uint64_t) offset;
  new_lwatime.second = lwatime.second + (offset - (uint64_t) offset);
  if( new_lwatime.second > 1 ) {
    new_lwatime.first += 1;
    new_lwatime.second -= 1;
  }
  if( new_lwatime.second < 0 ) {
    new_lwatime.first -= 1;
    new_lwatime.second += 1;
  }
  return new_lwatime;
}

inline double lwatime_diff(LWATime lwatime0, LWATime lwatime1) {
  double offset = lwatime0.first - lwatime1.first;
  offset += lwatime0.second - lwatime1.second;
  return offset;
}

inline time_t lwatime_to_time(LWATime lwatime) {
  return (time_t) lwatime.first + lwatime.second;
}

inline std::string lwatime_to_string(LWATime lwatime) {
  time_t start_t = lwatime_to_time(lwatime);
  struct tm *ptm = gmtime(&start_t);
  std::string output;
  output = std::to_string(ptm->tm_year + 1900)+"/"+std::to_string(ptm->tm_mon+1)+"/"+std::to_string(ptm->tm_mday) \
           +" "+std::to_string(ptm->tm_hour)+":"+std::to_string(ptm->tm_min)+":"+std::to_string(ptm->tm_sec + lwatime.second);
  return output;
}

/*
 Frequency conversion functions
*/

inline double tuningword_to_freq(uint32_t tuningword) {
  return (double) tuningword / ((uint64_t) 1<<32) * LWA_FS;
}

inline uint32_t freq_to_tuningword(double freq) {
  return (uint32_t) (freq / LWA_FS * ((uint64_t) 1<<32));
}

#endif // __INCLUDE_LWA_HPP_
