CC=		g++

OBJS=		strtokenizer.o dataset.o utils.o model_multi_thread.o
MAIN=		lda_multi_thread
 
all:	$(OBJS) $(MAIN).cpp
	$(CC) -o $(MAIN) $(MAIN).cpp $(OBJS) -lpthread
	strip $(MAIN)

strtokenizer.o:	strtokenizer.h strtokenizer.cpp
	$(CC) -c -o strtokenizer.o strtokenizer.cpp 

dataset.o:	dataset.h dataset.cpp
	$(CC) -c -o dataset.o dataset.cpp

utils.o:	utils.h utils.cpp
	$(CC) -c -o utils.o utils.cpp

model.o:	model.h model_multi_thread.cpp
	$(CC) -c -o model_multi_thread.o model_multi_thread.cpp  -L -pthread

test:
	

clean:
	rm $(OBJS) 
	rm $(MAIN)

