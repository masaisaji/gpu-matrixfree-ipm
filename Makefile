# ===== Compilers =====
CC   = gcc
NVCC = nvcc

# ===== Paths / Options =====
CUDA_HOME   ?= /usr/local/cuda

# ---- Optional Gurobi ----
USE_GUROBI  ?= 0
GUROBI_HOME ?= /opt/gurobi
GUROBI_VER  ?= 110   # e.g., 110 for 11.0.x, 95 for 9.5.x

SRCDIR = src
OBJDIR = build

# ===== Flags =====
# SANITIZE_FLAGS = -fsanitize=address,undefined,leak -fno-omit-frame-pointer

CFLAGS    = -Wall -O2 -g -Isrc -Ilib -I$(CUDA_HOME)/include
NVCCFLAGS = -O2 -g -Xcompiler -fPIC -Isrc -Ilib

LDFLAGS_COMMON = -llapacke -llapack -lblas -lglpk -lm -Llib \
                 -L$(CUDA_HOME)/lib64 \
                 -lcuda -lcudart -lcusparse -lcublas \
                 -Wl,-rpath,$(CUDA_HOME)/lib64
# $(SANITIZE_FLAGS)

# ===== Sources (Core, no 'main') =====
CORE_SRC = \
  $(SRCDIR)/lin_alg_helper.c \
  $(SRCDIR)/partial_cholesky.c \
  $(SRCDIR)/preconditioner.c \
  $(SRCDIR)/cg_solver.c \
  $(SRCDIR)/csr_utils.c \
  $(SRCDIR)/cuda_matvec.cu \
  $(SRCDIR)/cuda_matmul.cu \
  $(SRCDIR)/IPM.c \
  $(SRCDIR)/csr_IPM.c \
  $(SRCDIR)/mm_loader.c \
  $(SRCDIR)/solver_interface.c \
  lib/mmio.c \
  $(SRCDIR)/gen_initial_guess.c \
  $(SRCDIR)/csr_cg_solver.c

# ---- Optional Gurobi implementation vs stub
ifeq ($(USE_GUROBI),1)
  CORE_SRC += $(SRCDIR)/solve_w_gurobi.c
  CFLAGS   += -DUSE_GUROBI -I$(GUROBI_HOME)/include
  LDFLAGS  := $(LDFLAGS_COMMON) -L$(GUROBI_HOME)/lib -lgurobi$(GUROBI_VER) -Wl,-rpath,$(GUROBI_HOME)/lib
else
  CORE_SRC += $(SRCDIR)/solve_w_gurobi_stub.c
  LDFLAGS  := $(LDFLAGS_COMMON)
endif

# ===== App sources (with 'main') =====
# Default app
APP_SRC = $(CORE_SRC) $(SRCDIR)/testIPM.c
# Gurobi test app (only built when USE_GUROBI=1)
GUROBI_TEST_SRC = $(CORE_SRC) $(SRCDIR)/test_w_gurobi.c

# ===== Object lists =====
APP_COBJS        = $(patsubst %.c,$(OBJDIR)/%.o,$(filter %.c,$(APP_SRC)))
APP_CUOBJS       = $(patsubst %.cu,$(OBJDIR)/%.o,$(filter %.cu,$(APP_SRC)))
APP_OBJS         = $(APP_COBJS) $(APP_CUOBJS)

GUROBI_TEST_COBJS  = $(patsubst %.c,$(OBJDIR)/%.o,$(filter %.c,$(GUROBI_TEST_SRC)))
GUROBI_TEST_CUOBJS = $(patsubst %.cu,$(OBJDIR)/%.o,$(filter %.cu,$(GUROBI_TEST_SRC)))
GUROBI_TEST_OBJS   = $(GUROBI_TEST_COBJS) $(GUROBI_TEST_CUOBJS)

# ===== Targets =====
TARGET         = src/run
GUROBI_TARGET  = src/run_gurobi_test

# Build defaults
all: $(TARGET)
ifeq ($(USE_GUROBI),1)
all: $(GUROBI_TARGET)
endif

# ---- Link default app
$(TARGET): $(APP_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ---- Link Gurobi test app (only if enabled)
$(GUROBI_TARGET): $(GUROBI_TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ===== Compile rules =====
# Compile .c files with gcc
$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile .cu files with nvcc
$(OBJDIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ===== Utilities =====
clean:
	rm -rf $(OBJDIR) $(TARGET) $(GUROBI_TARGET)

.PHONY: all clean
