# Makefile for Kangaroo with Metal GPU support (macOS)
# Based on JeanLucPons/Kangaroo
#
# Usage:
#   make -f Makefile.metal          # Build with Metal support
#   make -f Makefile.metal clean    # Clean build files
#   make -f Makefile.metal test     # Run tests

# Compiler settings
CXX = clang++
OBJCXX = clang++
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

# Flags
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -DWITHMETAL -DWITHGPU
OBJCXXFLAGS = -O3 -std=c++17 -Wall -Wextra -DWITHMETAL -DWITHGPU -fobjc-arc
LDFLAGS = -framework Metal -framework Foundation -framework CoreGraphics

# Optional: Enable symmetry optimization
# Uncomment the following line to enable symmetry mode
# CXXFLAGS += -DUSE_SYMMETRY
# OBJCXXFLAGS += -DUSE_SYMMETRY

# Directories
SRCDIR = .
SECPDIR = SECPK1
METALDIR = Metal
OBJDIR = obj_metal

# Source files
SRCS = main.cpp \
       Kangaroo.cpp \
       Thread.cpp \
       HashTable.cpp \
       Timer.cpp \
       Check.cpp \
       Network.cpp \
       Backup.cpp \
       Merge.cpp \
       PartMerge.cpp

SECP_SRCS = $(SECPDIR)/Int.cpp \
            $(SECPDIR)/IntGroup.cpp \
            $(SECPDIR)/IntMod.cpp \
            $(SECPDIR)/Point.cpp \
            $(SECPDIR)/Random.cpp \
            $(SECPDIR)/SECP256K1.cpp

METAL_SRCS = $(METALDIR)/MetalEngine.mm

# Object files
OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(SRCS)))
SECP_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(SECP_SRCS)))
METAL_OBJS = $(patsubst %.mm,$(OBJDIR)/%.o,$(notdir $(METAL_SRCS)))

# Metal shader
METAL_SHADER = $(METALDIR)/KangarooKernel.metal
METAL_AIR = $(OBJDIR)/KangarooKernel.air
METAL_METALLIB = KangarooKernel.metallib

# Target
TARGET = KangarooMetal

# Default target
all: $(OBJDIR) $(METAL_METALLIB) $(TARGET)

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile Metal shader to AIR
$(METAL_AIR): $(METAL_SHADER) | $(OBJDIR)
	$(METAL) -c $(METAL_SHADER) -o $(METAL_AIR)

# Link AIR to metallib
$(METAL_METALLIB): $(METAL_AIR)
	$(METALLIB) $(METAL_AIR) -o $(METAL_METALLIB)

# Compile C++ sources
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile SECP256K1 sources
$(OBJDIR)/%.o: $(SECPDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile Metal engine (Objective-C++)
$(OBJDIR)/%.o: $(METALDIR)/%.mm | $(OBJDIR)
	$(OBJCXX) $(OBJCXXFLAGS) -c $< -o $@

# Link
$(TARGET): $(OBJS) $(SECP_OBJS) $(METAL_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Clean
clean:
	rm -rf $(OBJDIR)
	rm -f $(TARGET)
	rm -f $(METAL_METALLIB)
	rm -f *.air

# Test
test: $(TARGET)
	./$(TARGET) -t 1 -gpu

# Info
info:
	@echo "Kangaroo with Metal GPU support for macOS"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the application (default)"
	@echo "  clean   - Remove build files"
	@echo "  test    - Run a quick test"
	@echo "  info    - Show this message"
	@echo ""
	@echo "Options:"
	@echo "  To enable symmetry mode, uncomment USE_SYMMETRY in Makefile"

.PHONY: all clean test info
