#ifndef APO_TIMERS_H
#define APO_TIMERS_H

#include <time.h>
#include <sys/time.h>

double get_wall_time() {
  struct timeval time;
  if (gettimeofday(&time,NULL)){
      //  Handle error
      return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double get_cpu_time(){
  return (double)clock() / CLOCKS_PER_SEC;
}
    

#endif // APO_TIMERS_H
