nvcc -o test $1 -lcufft $(pkg-config --cflags --libs opencv4)
./test
