.PHONY: all clean

CC = gcc
CFLAGS = -g -Wall
SRC = $(wildcard *.c)
OBJ = $(SRC: %.c=%.o)
PROGS = $(SRC: %.c=proc_%)

all: $(OBJ) 
	echo $(SRC)
	echo $(OBJ)

%.o : %.c
	$(CC) $< -c $@


clean:
	@rm -rf *.o *.dSYM *.pyc *.swap proc* *.out
	@file * | grep executable | grep -v script | cut -d: -f 1 | xargs rm

