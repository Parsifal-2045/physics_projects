gif-h:
	git clone https://github.com/charlietangora/gif-h

serial: gif-h
	g++ circle_of_life.cpp -std=c++20 -o circle_of_life -O2 -I gif-h

clean:
	rm -rf circle_of_life circle_of_life_cuda

cuda:
	nvcc cuda_circle_of_life.cu -o cuda_circle_of_life -O2 -std=c++20 -I gif-h -ltbb -diag-suppress 550

tbb:
	g++ parallel_circle_of_life.cpp -std=c++20 -o parallel_circle_of_life -O2 -I gif-h -ltbb

all: serial cuda tbb

run: all
	./circle_of_life
	./cuda_circle_of_life
	./parallel_circle_of_life

