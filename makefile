CC        := g++ -O3

BASE_SRC  := $(wildcard base/*.cpp)
BASE_OBJ  := $(patsubst base/%.cpp,base/%.o,$(BASE_SRC))

EXAMPLE_SRC  := $(wildcard examples/*.cpp)
EXAMPLE_OBJ  := $(patsubst examples/%.cpp,examples/%.o,$(EXAMPLE_SRC))

vpath %.cpp base examples

all: jamon clean

jamon: $(BASE_OBJ) $(EXAMPLE_OBJ)
	$(CC) $^ -o $@ -lRandom

examples/%.o: %.cpp $(BASE_OBJ)
	$(CC) -Ibase -Iexamples -I/usr/local/include/Eigen -I/usr/local/include/RandomLib -c $< -o $@

base/%.o: %.cpp
	$(CC) -Ibase -I/usr/local/include/Eigen -c $< -o $@

clean:
	rm -rf base/*.o
	rm -rf examples/*.o
