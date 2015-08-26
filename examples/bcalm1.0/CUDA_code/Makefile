# .cu Objects that need to be compiled.
MAINOBJECTS = fdtd.cu DebugOutput.cu

ODIR=../obj
TARGET = $(ODIR)/fdtd
CFLAGS=--compile -L  /usr/local/hdf5_gcc/lib -lhdf5_hl -lhdf5 -lcudart -lfftw3 -lm -lcufft
#CFLAGS=--compile -L  /usr/local/hdf5_gcc/lib  -lhdf5 -lcudart -lfftw3 -lm
#CFLAGS=--compile -L -lhdf5_hl -lhdf5 -lcudart -lfftw3 -lm
LFLAGS=--link -L   /usr/local/hdf5_gcc/lib -lhdf5_hl -lhdf5 -lcudart /usr/lib/libfftw3.a -lm  -lcufft
#LFLAGS=--link -L   /usr/local/hdf5_gcc/lib  -lcudart /usr/lib/libfftw3.a -lm
#LFLAGS=--link -L -lhdf5_hl -lhdf5 -lcudart /usr/lib/libfftw3.a -lm
#LFLAGS=--link -L   /usr/local/hdf5_gcc/lib -lhdf5_hl -lhdf5 -lcudart -lfftw3 -lm

CXX=nvcc $(CFLAGS)
LINK=nvcc $(LFLAGS)


all:	$(TARGET)
	echo $(CXX)

emu:	
	make clean	
	make all "CXX=nvcc -deviceemu -g $(CFLAGS)" "LINK=nvcc -deviceemu -g $(LFLAGS)"

debug:	
	make clean
	make all "CXX=nvcc -g $(CFLAGS)" "LINK=nvcc -g $(LFLAGS)"

regall:	make clean
	make all "CXX=nvcc -maxrregcount=30 $(CFLAGS)" "LINK=nvcc  -maxrregcount=30 $(LFLAGS)"


# source files
DEPS := $(shell ls *.h *.cu)
SOURCESCU:=$(wildcard *.cu)
SOURCESCPP:=$(wildcard *.cpp)

# Build a list of objects to compile
_OBJCU = $(patsubst %.cu,%.o,$(MAINOBJECTS))
_OBJCPP = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJCU := $(patsubst %,$(ODIR)/%,$(_OBJCU))
OBJCPP := $(patsubst %,$(ODIR)/%,$(_OBJCPP))


.cu.o: $(DEPS)
	echo "Trying to build $@ from cu source $<"
	$(CXX) $< -o $@

.cpp.o: $(DEPS)
	echo "Trying to build $@ from cpp source $<"
	$(CXX) $< -o $@

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CXX) $< -o $@

$(ODIR)/%.o: %.cu $(DEPS)
	$(CXX) $< -o $@


$(TARGET): $(OBJCU) $(OBJCPP)
	$(LINK) $(OBJCU) $(OBJCPP) -o $@


TEST:
	echo $(OBJCU) $(OBJCPP)

clean:
	rm -f $(ODIR)/*.o *~
	rm -f $(TARGET)


########################
# BACKUP and SYNC functions
ARCHIVEDIR=../backup
BACKUPFILE := $(shell date +$(ARCHIVEDIR)/backup_MyCode_%Y%m%d_%H%M.tar.gz)
TARINCLUDE = *cu *cpp *h Makefile tar_exclude

tar:
	tar cvfzX $(BACKUPFILE) \
		tar_exclude \
		$(TARINCLUDE)
	cp $(BACKUPFILE) $(ARCHIVEDIR)/latest_backup.tar.gz
