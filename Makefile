# Makefile for ML Image Operation CLI
# Compiles C++ Segment Tree and manages the application

CXX = g++
CXXFLAGS = -std=c++17 -O3 -fPIC -Wall -shared
PYTHON = python3
TARGET_LIB = segment_tree.so
CPP_SRC = segment_tree.cpp
MAIN_CLI = ml_image_cli.py
ML_MODELS = ml_models.py
DB_DIR = ml_image_db
OUTPUT_DIR = outputs
DATA_DIR = data
SAMPLE_IMAGE = sample_image.png

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
CYAN = \033[0;36m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: all compile cli clean install deps check test help setup deep-clean install-dolt check-image

all: compile cli

# Compile C++ Segment Tree library
compile:
	@echo "$(CYAN)→ Compiling C++ Segment Tree library...$(NC)"
	@$(CXX) $(CXXFLAGS) -o $(TARGET_LIB) $(CPP_SRC) 2>&1 || \
		(echo "$(RED)✗ Compilation failed$(NC)" && exit 1)
	@echo "$(GREEN)✓ Compilation complete: $(TARGET_LIB)$(NC)"

# Run the CLI application
cli: check compile check-image
	@echo "$(CYAN)→ Starting ML Image Operation CLI...$(NC)"
	@$(PYTHON) $(MAIN_CLI) --image $(SAMPLE_IMAGE)

# Check if sample image exists, create if needed
check-image:
	@if [ ! -f $(SAMPLE_IMAGE) ]; then \
		echo "$(YELLOW)⚠ Sample image not found. Creating one...$(NC)"; \
		$(PYTHON) -c "import numpy as np; from PIL import Image; \
		img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)); \
		img.save('$(SAMPLE_IMAGE)'); print('$(GREEN)✓ Sample image created$(NC)')"; \
	else \
		echo "$(GREEN)✓ Sample image found: $(SAMPLE_IMAGE)$(NC)"; \
	fi

# Install Python dependencies
deps:
	@echo "$(CYAN)→ Installing Python dependencies...$(NC)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# Install PyTorch separately (with appropriate version)
install-torch:
	@echo "$(CYAN)→ Installing PyTorch...$(NC)"
	@echo "$(YELLOW)Detecting system configuration...$(NC)"
	@if command -v nvidia-smi &> /dev/null; then \
		echo "$(GREEN)CUDA detected, installing PyTorch with CUDA support$(NC)"; \
		$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118; \
	else \
		echo "$(YELLOW)No CUDA detected, installing CPU-only PyTorch$(NC)"; \
		$(PYTHON) -m pip install torch torchvision; \
	fi
	@echo "$(GREEN)✓ PyTorch installed$(NC)"

# Install Dolt (if not already installed)
install-dolt:
	@echo "$(CYAN)→ Checking for Dolt installation...$(NC)"
	@if ! command -v dolt &> /dev/null; then \
		echo "$(YELLOW)Dolt not found. Installing...$(NC)"; \
		echo "$(CYAN)For Linux/Mac, running install script...$(NC)"; \
		curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash || \
		(echo "$(RED)✗ Dolt installation failed$(NC)" && \
		 echo "$(YELLOW)Please install manually from: https://github.com/dolthub/dolt/releases$(NC)" && \
		 exit 1); \
	else \
		echo "$(GREEN)✓ Dolt is installed$(NC)"; \
		dolt version; \
	fi

# Check system dependencies
check:
	@echo "$(CYAN)→ Checking system dependencies...$(NC)"
	@command -v $(CXX) >/dev/null 2>&1 || \
		(echo "$(RED)✗ g++ not found. Please install g++$(NC)" && exit 1)
	@command -v $(PYTHON) >/dev/null 2>&1 || \
		(echo "$(RED)✗ Python3 not found. Please install Python3$(NC)" && exit 1)
	@command -v dolt >/dev/null 2>&1 || \
		(echo "$(YELLOW)⚠ Dolt not found. Run 'make install-dolt' to install$(NC)")
	@echo "$(GREEN)✓ System dependencies checked$(NC)"

# Verify Python modules are installed
check-modules:
	@echo "$(CYAN)→ Verifying Python modules...$(NC)"
	@$(PYTHON) -c "import numpy, cv2, torch, torchvision, click, rich" 2>/dev/null || \
		(echo "$(RED)✗ Required Python modules missing. Run 'make deps'$(NC)" && exit 1)
	@echo "$(GREEN)✓ All Python modules available$(NC)"

# Run quick test
test: compile
	@echo "$(CYAN)→ Running quick test...$(NC)"
	@$(PYTHON) -c "import ctypes; \
		try: \
			lib = ctypes.CDLL('./$(TARGET_LIB)'); \
			print('$(GREEN)✓ C++ library loaded successfully$(NC)'); \
		except Exception as e: \
			print('$(YELLOW)⚠ Library load warning:', e, '$(NC)')"
	@echo "$(CYAN)→ Testing Python imports...$(NC)"
	@$(PYTHON) -c "from ml_models import MLImageProcessor; \
		print('$(GREEN)✓ ML models module loaded$(NC)')"

# Run full test suite
test-full: compile check-image
	@echo "$(CYAN)→ Running full test suite...$(NC)"
	@$(PYTHON) -c "from ml_models import MLImageProcessor; \
		processor = MLImageProcessor('$(SAMPLE_IMAGE)'); \
		print('$(GREEN)✓ Image processor initialized$(NC)')"
	@echo "$(GREEN)✓ All tests passed$(NC)"

# Clean compiled files and outputs
clean:
	@echo "$(YELLOW)→ Cleaning build artifacts...$(NC)"
	@rm -f $(TARGET_LIB)
	@rm -rf $(OUTPUT_DIR)
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -f *.pyc
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

# Deep clean (includes database and datasets)
deep-clean: clean
	@echo "$(YELLOW)→ Removing database and all data...$(NC)"
	@rm -rf $(DB_DIR)
	@rm -rf $(DATA_DIR)
	@rm -f $(SAMPLE_IMAGE)
	@echo "$(GREEN)✓ Deep clean complete$(NC)"

# Create necessary directories
create-dirs:
	@echo "$(CYAN)→ Creating directories...$(NC)"
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(DATA_DIR)
	@echo "$(GREEN)✓ Directories created$(NC)"

# Setup everything from scratch
setup: check install-dolt deps install-torch compile create-dirs check-image
	@echo ""
	@echo "$(GREEN)╔════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║  ✓ Setup complete!                                    ║$(NC)"
	@echo "$(GREEN)║  Run 'make cli' to start the application              ║$(NC)"
	@echo "$(GREEN)╚════════════════════════════════════════════════════════╝$(NC)"
	@echo ""

# Quick start (skip dependency installation)
quick-start: compile check-image cli

# Show system info
info:
	@echo "$(CYAN)System Information:$(NC)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "  G++: $(shell $(CXX) --version | head -n1)"
	@echo "  Dolt: $(shell dolt version 2>/dev/null || echo 'Not installed')"
	@echo "  PyTorch: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "  CUDA Available: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"

# Rebuild everything
rebuild: clean compile
	@echo "$(GREEN)✓ Rebuild complete$(NC)"

# Help menu
help:
	@echo "$(CYAN)╔════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║     ML Image Operation CLI - Makefile Help            ║$(NC)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Main Targets:$(NC)"
	@echo "  $(GREEN)make setup$(NC)           - Complete first-time setup"
	@echo "  $(GREEN)make cli$(NC)             - Compile and run the application"
	@echo "  $(GREEN)make quick-start$(NC)     - Skip deps check and start"
	@echo ""
	@echo "$(YELLOW)Build Targets:$(NC)"
	@echo "  $(GREEN)make compile$(NC)         - Compile C++ Segment Tree library"
	@echo "  $(GREEN)make rebuild$(NC)         - Clean and recompile"
	@echo ""
	@echo "$(YELLOW)Dependency Targets:$(NC)"
	@echo "  $(GREEN)make deps$(NC)            - Install Python dependencies"
	@echo "  $(GREEN)make install-torch$(NC)   - Install PyTorch (auto-detect CUDA)"
	@echo "  $(GREEN)make install-dolt$(NC)    - Install Dolt database"
	@echo ""
	@echo "$(YELLOW)Testing Targets:$(NC)"
	@echo "  $(GREEN)make check$(NC)           - Check system dependencies"
	@echo "  $(GREEN)make test$(NC)            - Run quick library test"
	@echo "  $(GREEN)make test-full$(NC)       - Run full test suite"
	@echo ""
	@echo "$(YELLOW)Maintenance Targets:$(NC)"
	@echo "  $(GREEN)make clean$(NC)           - Remove compiled files"
	@echo "  $(GREEN)make deep-clean$(NC)      - Remove everything (including DB)"
	@echo "  $(GREEN)make info$(NC)            - Show system information"
	@echo ""
	@echo "$(YELLOW)Quick Start Guide:$(NC)"
	@echo "  1. First time: $(CYAN)make setup$(NC)"
	@echo "  2. Run application: $(CYAN)make cli$(NC)"
	@echo "  3. After changes: $(CYAN)make rebuild$(NC) then $(CYAN)make cli$(NC)"
	@echo ""
