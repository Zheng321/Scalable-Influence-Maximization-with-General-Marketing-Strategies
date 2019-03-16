XX = g++
CFLAGS = -g 
CLIBS = -lpthread

INCLUDE = $(wildcard ./*.h)
SOURCES = $(wildcard ./*.cpp)
INCLUDE_DIRS = -I./*.h

TARGET = nodeSelect

OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))

$(TARGET) : $(OBJECTS)
	$(XX)  -std=c++11 $(CFLAGS) $^ -o $@ $(CLIBS)
$(OBJECTS) : %.o : %.cpp 
	$(XX)  -std=c++11 -c $(CFLAGS) $< -o $@ $(INCLUDE_DIRS)
