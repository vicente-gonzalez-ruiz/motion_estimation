## Makefile
#  It helps build a software from its source files, a way to organize
#  code, and its compilation and linking.

# The MCTF project has been supported by the Junta de Andalucía through
# the Proyecto Motriz "Codificación de Vídeo Escalable y su Streaming
# sobre Internet" (P10-TIC-6548).

default:	all

M4 := $(shell which m4)

#ifeq ($(M4),)
#$(warning No m4 found!)
#false
#endif

ifeq ($(shell which g++),)
$(warning No c++ found :-/)
false
endif

CC = g++

BIN = $(HOME)/bin
MCTF_BIN = $(MCTF)/bin
EXE =
#CFLAGS = -g
#CFLAGS = -O3 -pipe
#GCC_FLAGS = -O3 -pipe
GCC_FLAGS = -g
#CPP_FLAGS = -O3 -pipe
CPP_FLAGS = -g
DEFS =
GCC_FLAGS += $(DEFS)
CPP_FLAGS += $(DEFS)

# Rules.
$(BIN)/% :: %.c
	gcc $(GCC_FLAGS) $< -o $@ -lm

$(MCTF_BIN)/% :: %.c
	gcc $(GCC_FLAGS) $< -o $@ -lm

$(BIN)/% :: %.cpp
	g++ $(CPP_FLAGS) $< -o $@ -lm

$(MCTF_BIN)/% :: %.cpp
	g++ $(CPP_FLAGS) $< -o $@ -lm

PYs := $(willcard *.py)

TMPs =
TMPs += $(PYs:%.py=$(BIN)/%)
#$(BIN)/%:	%.py
#	(echo "changequote({{,}})dnl"; cat $*.py) | m4 $(DEFS) > $@; chmod +x $@
$(BIN)/%:	%.py
	cp $*.py $@; chmod +x $@

$(MCTF_BIN)/%:	%.py
	cp $*.py $@; chmod +x $@

$(BIN)/%.py:	%.py
	cp $*.py $@; chmod +x $@

$(MCTF_BIN)/%.py:	%.py
	cp $*.py $@; chmod +x $@

$(BIN)/%:	%.sh
#	(echo "changequote({{,}})dnl"; cat $<) | m4 $(DEFS) > $@; chmod +x $@
#	m4 $(DEFS) < $< > $@; chmod +x $@
	cp $*.sh $@; chmod +x $@

$(MCTF_BIN)/%:	%.sh
	cp $*.sh $@; chmod +x $@

#CPPM4s := $(wildcard *.cpp.m4)

#TMPs =
#TMPs += $(CPPM4s:%.cpp.m4=%.cpp)
#%.cpp:	%.cpp.m4
#	(echo "changequote(\`[[[', \`]]]')"; cat $*.cpp.m4) | m4 $(DEFS) > $*.cpp

$(MCTF_BIN)/motion_estimate:	motion_estimate.cpp display.cpp Haar.cpp 5_3.cpp dwt2d.cpp texture.cpp motion.cpp common.h
EXE += $(MCTF_BIN)/motion_estimate

all:	$(EXE)

objetives:
	@echo $(EXE) all clean

info:	objetives

clean:
	rm -f $(EXE) ../bin/*.pyc ../bin/*.py $(TMPs)
