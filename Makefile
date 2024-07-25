# Variables
BIN_DIR := bin
OUTPUT_EXEC := $(BIN_DIR)/sorting_rush

# Default target
all: run

# Run target
run: $(OUTPUT_EXEC)
	./$(OUTPUT_EXEC)

# Clean target
clean:
	rm -f $(DATA_DIR)/output.txt
	rm -f $(OUTPUT_EXEC)

.PHONY: all run clean
