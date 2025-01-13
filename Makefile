TARGET=a.out
CXX=g++
LD=g++
OPENCV=-I/usr/include/opencv4 -g
CXXFLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_face -lopencv_objdetect -lopencv_dnn -std=c++11
#CXXFLAGS=`pkg-config --cflags --libs opencv4`
all:
	$(CXX) $(OPENCV) main.cpp $(CXXFLAGS)
	@./$(TARGET)
