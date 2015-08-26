#ifndef PRINT_LOG

#include "PrintLog.h"

void PrintLog(const char *msg, ...) {
  va_list argp;
  va_start(argp, msg);
  vfprintf(stdout, msg, argp);
  va_end(argp);

  va_start(argp, msg);
  vfprintf(simout, msg, argp);
  va_end(argp);

}

void PrintLogFile(const char *msg, ...) {
  va_list argp;
  va_start(argp, msg);
  vfprintf(simout, msg, argp);
  va_end(argp);
}

#endif//printlog