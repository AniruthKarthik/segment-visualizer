# AI-SEG: ML Image Segmentation CLI

AI-SEG is a powerful, menu-driven command-line interface for machine learning-based image segmentation and analysis. It features hyperparameter optimization, versioned data storage with Dolt, and high-performance metric querying using a C++ Segment Tree.

## 🚀 Features

- **Three ML Models**: Choose between a Simple Autoencoder, a Deep Autoencoder with residual connections, and a U-Net for different levels of quality and performance.
- **Hyperparameter Optimization**: Automatically test numerous combinations of hyperparameters to find the best model for your data.
- **DoltDB Integration**: All experiment results are versioned in a Dolt database, allowing you to track, query, and revert experiments with Git-like commands.
- **High-Speed Queries**: A C++ Segment Tree backend provides fast (O(log n)) queries for metrics like top-K results and range-based searches.
- **Interactive CLI**: A user-friendly, menu-driven interface built with `rich` for a modern and intuitive experience.
- **Advanced Analytics**: Visualize results, compare model performance, and query the database with custom SQL.
- **Extensible**: The modular design makes it easy to add new models, metrics, and functionality.

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- g++ with C++17 support
- Dolt database (https://github.com/dolthub/dolt)
- Make

### Python Dependencies
The required Python packages are listed in `requirements.txt`.

## 🔧 Installation

A `Makefile` is provided for easy installation and setup.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd hyperparameter-analyzer
    ```

2.  **Run the setup command:**
    This will install all dependencies, compile the C++ library, and prepare the environment.
    ```bash
    make setup
    ```

    This command will:
    - Install Dolt.
    - Install Python dependencies from `requirements.txt`.
    - Install PyTorch (with CUDA support if available).
    - Compile the C++ Segment Tree library.
    - Create necessary directories.

## 🎯 Usage

### Start the Application
To start the interactive CLI, run:
```bash
make cli
```

You can also specify a custom image:
```bash
python3 ml_image_cli.py --image /path/to/your/image.png
```

### Interactive Menu
The main menu provides access to all of the CLI's features:

```
ML Image CLI - Main Menu
======================================================================
[bold cyan]Training & Evaluation[/bold cyan]
  [1]  Run Hyperparameter Training
  [2]  Export Best Model

[bold cyan]Segment Tree Queries[/bold cyan]
  [3]  Query Top-K (Segment Tree)
  [4]  Query Max PSNR in LR Range
  [5]  Find Configs Above Threshold
  [6]  Analyze Percentiles

[bold cyan]Database Queries[/bold cyan]
  [7]  Query Top-K (Database)
  [8]  Compare Modes
  [9]  Execute Custom SQL

[bold cyan]Analysis & Visualization[/bold cyan]
  [10] Visualize Results
  [11] Show Statistics

[bold cyan]System[/bold cyan]
  [12] Exit
```

## 🏗️ Project Structure

```
/home/ani/aiseg/
├───.gitignore
├───Makefile                  # For automating build, run, and test processes
├───ml_image_cli.py           # The main CLI application
├───ml_models.py              # Contains the ML model definitions
├───readme.md                 # This file
├───requirements.txt          # Python dependencies
├───sample_image.png          # A sample image for testing
├───segment_tree.cpp          # C++ implementation of the Segment Tree
├───ml_image_db/              # DoltDB database directory
└───outputs/                  # Directory for output images and models
```

## 🧠 ML Models

The CLI includes three different models for image denoising/segmentation:

1.  **SimpleAutoencoder**: A basic model for quick tests.
2.  **DeepAutoencoder**: A more complex model with residual connections for better performance.
3.  **UNetAutoencoder**: A U-Net-based model for the highest quality results.

Each model is trained on the CIFAR-10 dataset, and the hyperparameters can be tuned through the CLI.

## 📊 Database Schema

The Dolt database stores results in three tables, one for each mode: `testdb`, `realdb`, and `advanceddb`. The schema for `realdb` is as follows:

```sql
CREATE TABLE realdb (
    id INT PRIMARY KEY AUTO_INCREMENT,
    learning_rate DOUBLE,
    momentum DOUBLE,
    kernel_size INT,
    stride INT,
    epochs INT,
    batch_size INT,
    dropout_rate DOUBLE,
    weight_decay DOUBLE,
    loss DOUBLE,
    psnr DOUBLE,
    ssim DOUBLE,
    mae DOUBLE,
    runtime DOUBLE,
    model_type VARCHAR(50),
    timestamp DATETIME
);
```

## 🛠️ Makefile Commands

| Command | Description |
|---|---|
| `make setup` | Complete first-time setup. |
| `make cli` | Compile and run the application. |
| `make compile` | Compile the C++ Segment Tree library. |
| `make deps` | Install Python dependencies. |
| `make test` | Run quick tests. |
| `make clean` | Remove compiled files and outputs. |
| `make deep-clean` | Remove all generated files, including the database. |
| `make help` | Show the help menu. |

## 🐛 Troubleshooting

- **C++ Library Not Found**: If you get an error about `segment_tree.so` not being found, run `make compile` to build it.
- **Dolt Not Installed**: Run `make install-dolt` to install Dolt.
- **Missing Python Dependencies**: Run `make deps` to install the required Python packages.
