#ifndef _GETFILEARGS_H_
#define _GETFILEARGS_H_

#include <stdio.h>

#define PARAM_CHAR          0
#define PARAM_INT           1
#define PARAM_LONG          2
#define PARAM_FLOAT         3
#define PARAM_STRING        4
#define PARAM_DIVIDER     100
#define PARAM_END         101

typedef char *GetValueDescription(int n);

typedef struct {
  char type;
  char *identifier;
  void *where;
  char *defValue;
  char *description;
  GetValueDescription *getValueDescription;
} ParamStruct;

int getFirstString(FILE *f, char *s);
int setParamValue(FILE *f, ParamStruct *param);
int getParamsFromFile(char *filename, ParamStruct params[]);
int printParamsDescription(FILE *out, ParamStruct params[]);
int printParamValues(FILE *out, ParamStruct params[]);

char *yesNoDescriptor(int i);

#endif
