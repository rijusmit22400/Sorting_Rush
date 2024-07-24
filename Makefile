CC = nvcc
CFLAGS = -I./lib -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

SRC_DIR = src
BIN_DIR = bin
LIB_DIR = lib

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(BIN_DIR)/%.o)
EXEC = $(BIN_DIR)/grayscale_converter

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_DIR)

run: $(EXEC)
	./$(EXEC)