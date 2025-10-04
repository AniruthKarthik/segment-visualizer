cli:
	@echo "🔧 Building Segment Tree binary..."
	g++ Segment.cpp -o build/segment -O2
	@echo "🚀 Launching CLI..."
	python3 cli.py

build:
	g++ Segment.cpp -o build/segment -O2
	@echo "✅ Build complete"

clean:
	rm -rf build/*
	@echo "🧹 Cleaned all build artifacts."
