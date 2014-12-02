// ################################################################################
//
// name:          getFileArgs.cc      
//
// author:        Mohamed Wahib
//
// purpose:       functions for reading the input file, printing the description 
//                of the parameters that can be processed, and the related
//
// last modified: Feb 2014
//
// ################################################################################

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "getFileArgs.h"
#include "memalloc.h"
#include "utils.h"

// ---------------------------------------
// some compilers do not know strdup, I do
// ---------------------------------------

#ifndef strdup
#define strdup(s) (char*) strcpy((char*) Malloc((long)strlen(s)+1),s)
#endif

// -----------------------------------
// the type description for parameters
// -----------------------------------

char *typeDesc[30] = 
        { "char", 
	  "int", 
	  "long", 
	  "float", 
	  "char*",
	  "(int,int)*"
	};

// -------------
// no comment...
// -------------

char *yesString = "Yes";
char *noString = "No";

// ================================================================================
//
// name:          yesNoDescriptor
//
// function:      returns "Yes" if input is true, "No" otherwise
//
// parameters:    i............a number determining what to return
//
// returns:       (char*) resulting string ("Yes"/"No")
//
// ================================================================================

char *yesNoDescriptor(int i)
{
  if (i)
    return yesString;
  else
    return noString;
}

// ================================================================================
//
// name:          getFirstString
//
// function:      gets a first string composed of letters or characters "-" and "_"
//                from a current file position and stores it; also stores the first
//                character of the remaining output; skips spaces at the beginning
//                if needed
//
// parameters:    f............the pointer to input stream
//                s............the pointer where to store the read string
//                restC........the character following the string
//
// returns:       (int) 0 if end of file reached, non-zero otherwise
//
// ================================================================================

int getFirstString(FILE *f, char *s, char *restC)
{
  register int i;
  char c;

  do 
  {
    c = fgetc(f);
  } while ((!feof(f)) && 
           (((c<'a') || (c>'z')) &&
            ((c<'A') || (c>'Z')) &&
            ((c<'0') || (c>'9')) &&
            (c!='_') &&
	    (c!='-')));
  if (!feof(f))
  {
    i=0;
    while ((c!=10) && (!feof(f)) && (c!=' ') && (c!='='))
    {
      s[i++] = c;
      c = fgetc(f);
    }
    s[i]=0;
  }

  if (restC)
    *restC=c;

  return (!feof(f));
}

// ================================================================================
//
// name:          setParamValue
//
// function:      sets the value of a parameter from the file, the pointer in the
//                file must be at the beginning of the value or spaces before it;
//                if the pointer to the file is NULL, the value of a paramter
//                is set to its default value
//
// parameters:    f............the pointer to the input stream
//                param........the pointer to the parameter to set
//
// returns:       (int) 0 (if everything went fine)
//
// ================================================================================

int setParamValue(FILE *f, ParamStruct *param)
{
  char s[100];
  int iTmp;
  time_t t;

  // set the value of paremeter from the file, if the file is NULL then the value is chosen
  // to be the default value given as the param's attribute

  switch (param->type) 
    {
    case PARAM_CHAR:
      if (f)
	getFirstString(f,s,NULL);
      else
	strcpy(s,param->defValue);

      sscanf(s,"%i",&iTmp);

      *(( char*)param->where)=(char)iTmp;

      break;
      
    case PARAM_INT: 
      if (f)
	getFirstString(f,s,NULL);
      else
	strcpy(s,param->defValue);

      if (strcmp(s,"time"))
	sscanf(s,"%i",(int*)param->where);
      else
	*((int*)param->where) = time(&t);

      break;
      
    case PARAM_LONG: 
      if (f)
	getFirstString(f,s,NULL);
      else
	strcpy(s,param->defValue);

      if (strcmp(s,"time"))
	sscanf(s,"%li",(long*)param->where);
      else
	*((long*)param->where) = time(&t);

      sscanf(s,"%li",(long *)param->where);

      break;
      
    case PARAM_FLOAT:
      if (f)
	getFirstString(f,s,NULL);
      else
	strcpy(s,param->defValue);

      sscanf(s,"%f",(float*)param->where);
      break;

    case PARAM_STRING:
      if (f)
	{
	  getFirstString(f,(char*) s,NULL);
	  *((char**)param->where) = strdup((char*)s);
	}
      else
	{
	  if (param->defValue)
	    *((char**)param->where) = (char*) strdup((char*)param->defValue);
	  else
	    *((char**)param->where) = (char*) NULL;
	}
      break;

    case PARAM_DIVIDER:
      break;

    case PARAM_END:
      break;

    default: 
      fprintf(stderr,"ERROR: Undefined parameter type (%u)\n",param->type);
      exit(-2);
    }

  // get back

  return 0;
}

// ================================================================================
//
// name:          getParamsFromFile
//
// function:      reads the values of parameters from a file
//
// parameters:    filename.....the name of an input file
//                params.......the array of parameters to be set (either from the
//                             file or default values
//
// returns:       (int) 0 (if everything went fine)
//
// ================================================================================

int getParamsFromFile(char *filename, ParamStruct params[])
{
  FILE *f=NULL;
  char s[200];
  register int i;
  int numParams;
  char *defined;
  int which;
  char c;

  numParams=0;
  while (params[numParams].type!=PARAM_END) numParams++;

  // allocate memory for the array saying what was defined and what hasn't been defined yet

  defined = (char*) Malloc(numParams);

  // set this array values to 0 (nothing has been defined yet)

  for (i=0; i<numParams; i++)
      defined[i]=0;

  // if there's configuration file name given, this file has got to exist

  if (filename)
    {
      f = fopen(filename,"r");

      if (f==NULL)
	{
	  fprintf(stderr,"ERROR: File %s doesn't exist!\n",filename);
	  exit(-1);
	}
    }

  // read configuration file...

  if (f)
  while (!feof(f))
  {
    // if it is possible, read the first identifier

    if (getFirstString(f,s,&c))
    {
          which=-1;
          for (i=0; (i<numParams) && (which==-1); i++)
	      if ((params[i].type!=PARAM_DIVIDER) && (!strcmp(params[i].identifier,s)))
                 which=i;

	  // does identifier not exist?

          if (which==-1)
	    {
	      fprintf(stderr,"ERROR: Parameter %s in file %s is not understood!\n",s,filename);
	      exit(-1);
	    }

	  // defined twice?

          if (defined[which])
	    {
              fprintf(stderr,"ERROR: Parameter %s in file %s was redefined!\n",s,filename);
	      exit(-1);
	    }

	  // missing '=' after identifier?

          while ((!feof(f)) && (c!='=')) c = fgetc(f);
          if ((feof(f))||(c!='='))
	    {
	      fprintf(stderr,"ERROR: Parameter identifier %s in file %s is not followed by '='!\n",params[which].identifier,filename);
	      exit(-1);
	    }

	  // read the parameter value 

          setParamValue(f,&(params[which]));

          defined[which]=1;
    }
  }

  // set default values for the rest of parameters (that has not been define in configuration
  // file)

  for (i=0; i<numParams; i++)
  if (!defined[i])
    setParamValue(NULL,&(params[i]));

  // free allocated memory

  free(defined);

  // get back

  return 0;
}

// ================================================================================
//
// name:          printParamsDescription
//
// function:      prints out the description of parameters that can be specified
//
// parameters:    out..........where to print the description
//                params.......the array of parameters to print the description of
//
// returns:       (int) 0 (if everything went fine)
//
// ================================================================================

int printParamsDescription(FILE *out, ParamStruct params[])
{
  int i=0;

  // print the header (description of information pieces that follow)

  printf("Configuration file description:\n\n");
  fputs("Description                                                Identifier                 Type       Default\n",out);
  printf("----------------------------------------------------------------------------------------------------------------\n");

  // print the description of all parameters

  while (params[i].type!=PARAM_END)
    {
      if (params[i].type==PARAM_DIVIDER)
	printf("\n");
      else
	fprintf(out,"%-58s %-26s %-10s %-8s\n",params[i].description,params[i].identifier, typeDesc[params[i].type], (params[i].defValue)? params[i].defValue:"(null)"); 
	
      i++;
    }

  // get back

  return 0;
}

// ================================================================================
//
// name:          printParamsDescription
//
// function:      prints out the values of the parameters
//
// parameters:    out..........where to print the values to
//                params.......the array of parameters to print the values of
//
// returns:       (int) 0 (if everything went fine)
//
// ================================================================================

int printParamValues(FILE *out, ParamStruct params[])
{

  int i=0;

  // is output stream null

  if (out==NULL)
    return 0;

  // print out the header (the description of information that follows)

  fprintf(out,"Parameter Values:\n\n");
  fputs("Description                                                Identifier                 Type       Value\n",out);
  fprintf(out,"----------------------------------------------------------------------------------------------------------------\n");

  // print out the descriptions and the values of all parameters

  while (params[i].type!=PARAM_END)
    {
      if (params[i].type!=PARAM_DIVIDER)
	fprintf(out,"%-58s %-26s %-10s ",params[i].description,params[i].identifier, typeDesc[params[i].type]);

      switch (params[i].type)
	{
	case PARAM_CHAR:
	  if (params[i].getValueDescription)
	    {
	      fprintf(out,"%i (",*((char*)params[i].where));
	      fputs((*((GetValueDescription*)params[i].getValueDescription))(*((char*)params[i].where)),out);
	      fprintf(out,")");
	    }
	  else
	    fprintf(out,"%i",*((char*)params[i].where));
	  break;

	case PARAM_INT:
	  if (params[i].getValueDescription)
	    {
	      fprintf(out,"%i (",*((char*)params[i].where));
	      fputs((*((GetValueDescription*)params[i].getValueDescription))(*((int*)params[i].where)),out);
	      fprintf(out,")");
	    }
	  else
	    fprintf(out,"%i",*((int*)params[i].where));
	  break;

	case PARAM_LONG:
	  fprintf(out,"%li",*((long*)params[i].where));
	  break;

	case PARAM_FLOAT:
	  fprintf(out,"%f",*((float*)params[i].where));
	  break;

	case PARAM_STRING:
	  if (*((char**)params[i].where))
	    fputs(*((char**)params[i].where),out);
	  else
	    fputs("(null)",out);
	  break;
	  
	case PARAM_DIVIDER:
	  break;
	}

      fprintf(out,"\n");

      i++;
    }

  // get back
  
  return 0;
}
