CCOMPILE=mpic++
CPPFLAGS= -I$(HADOOP_HOME)/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I ../system -Wno-deprecated -O2
LIB = -L$(HADOOP_HOME)/lib/native -L$(JAVA_HOME)/jre/lib/amd64/server
LDFLAGS = -lhdfs -ljvm

SRC = run.cpp \
      ../system/basic/metrics_logger.cpp

all: run

run: $(SRC)
	$(CCOMPILE) $(SRC) $(CPPFLAGS) $(LIB) $(LDFLAGS) -o run

clean:
	rm -f run
