# Makefile for metal_infer — Pure C/Metal MoE inference engine
#
# Targets:
#   make          — build metal_infer binary (default: Qwen3.5-397B)
#   make MODEL=qwen3-coder — build for Qwen3-Coder-480B
#   make infer    — build inference engine
#   make chat     — build interactive chat TUI
#   make clean    — remove build artifacts
#
# Model selection:
#   MODEL=qwen35-397b    (default) Qwen3.5-397B-A17B-4bit
#   MODEL=qwen3-coder    Qwen3-Coder-480B-A35B-Instruct-4bit
#
# Note: Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

# Model selection via MODEL= variable
ifeq ($(MODEL),qwen3-coder)
  MODEL_FLAG = -DMODEL_QWEN3_CODER_480B
else ifeq ($(MODEL),qwen3.6-35b-a3b)
  MODEL_FLAG = -DMODEL_QWEN36_35B_A3B
else
  MODEL_FLAG =
endif

TARGET = metal_infer
MAIN_SRC = main.m

# Optional: offline shader compilation (faster startup, but not required)
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = shaders.metal
SHADER_AIR = shaders.air
SHADER_LIB = shaders.metallib

# Inference engine (complete forward pass)
INFER_TARGET = infer
INFER_SRC = infer.m

# Chat TUI (interactive multi-turn)
CHAT_TARGET = chat
CHAT_SRC = chat.m

.PHONY: all clean run verify bench moe moebench full fullbench fast metallib infer infer-run chat-run build-chat

all: $(TARGET) $(INFER_TARGET)

# Build the binary (shaders compiled at runtime from source)
$(TARGET): $(MAIN_SRC) $(SHADER_SRC) model_config.h
	$(CC) $(CFLAGS) $(MODEL_FLAG) $(FRAMEWORKS) $(LDFLAGS) $(MAIN_SRC) -o $(TARGET)

# Optional: pre-compile shaders (not required — runtime compilation is the default)
metallib: $(SHADER_LIB)

$(SHADER_AIR): $(SHADER_SRC)
	$(METALC) -c $(SHADER_SRC) -o $(SHADER_AIR)

$(SHADER_LIB): $(SHADER_AIR)
	$(METALLIB_TOOL) $(SHADER_AIR) -o $(SHADER_LIB)

# Build the inference engine
$(INFER_TARGET): $(INFER_SRC) model_config.h
	$(CC) $(CFLAGS) $(MODEL_FLAG) $(FRAMEWORKS) $(LDFLAGS) $(INFER_SRC) -o $(INFER_TARGET)

# Build the chat client (thin HTTP/SSE client + linenoise line editor)
$(CHAT_TARGET): $(CHAT_SRC) linenoise.c linenoise.h model_config.h
	$(CC) -O2 -Wall -fobjc-arc $(MODEL_FLAG) -framework Foundation $(CHAT_SRC) linenoise.c -o $(CHAT_TARGET)

clean:
	rm -f $(TARGET) $(INFER_TARGET) $(CHAT_TARGET) $(SHADER_AIR) $(SHADER_LIB)

# Run targets
run: $(TARGET)
	./$(TARGET) --layer 0 --expert 0

verify: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --verify

fast: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --verify

bench: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --benchmark

moe: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe

moebench: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe --benchmark

full: $(TARGET)
	./$(TARGET) --fast --full --k 4

fullbench: $(TARGET)
	./$(TARGET) --fast --full --k 4 --benchmark

# Inference engine targets
build-infer: $(INFER_TARGET)

infer-run: $(INFER_TARGET)
	./$(INFER_TARGET) --prompt "Hello, what is" --tokens 20 --k 4

# Chat TUI targets (use: make chat)

chat-run: $(CHAT_TARGET)
	./$(CHAT_TARGET) --k 4
