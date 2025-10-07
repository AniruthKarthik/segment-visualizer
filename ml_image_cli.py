#!/usr/bin/env python3
"""
Enhanced ML Image Operation CLI with Advanced Segment Tree Integration
Supports persistent data storage, multiple architectures, and comprehensive analytics
"""

import os
import sys
import json
import time
import ctypes
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.tree import Tree
from rich import box
import subprocess
import bisect
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from ml_models import MLImageProcessor

# Initialize Rich console
console = Console()


class SegmentTreeWrapper:
    """Enhanced Segment Tree wrapper with multiple operations"""
    
    def __init__(self, lib_path="./segment_tree.so"):
        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_functions()
            self.tree_ptr = None
            self.values = []
            console.print("[green]✓ Segment Tree library loaded[/green]")
        except OSError as e:
            console.print(f"[yellow]⚠ Warning: C++ library not found, using Python fallback[/yellow]")
            self.lib = None
            self.values = []
    
    def _setup_functions(self):
        if not self.lib:
            return
        
        self.lib.create_segment_tree.restype = ctypes.c_void_p
        self.lib.create_segment_tree.argtypes = [ctypes.c_int]
        self.lib.update_value.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.lib.query_range_max.restype = ctypes.c_double
        self.lib.query_range_max.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.query_top_k.restype = ctypes.POINTER(ctypes.c_int)
        self.lib.query_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.destroy_segment_tree.argtypes = [ctypes.c_void_p]
    
    def create(self, size: int):
        if self.lib:
            self.tree_ptr = self.lib.create_segment_tree(size)
        self.values = [0.0] * size
        console.print(f"[cyan]→ Created Segment Tree (size={size})[/cyan]")
    
    def update(self, index: int, value: float):
        if index < len(self.values):
            self.values[index] = value
        if self.lib and self.tree_ptr:
            self.lib.update_value(self.tree_ptr, index, value)
    
    def query_top_k(self, k: int) -> List[int]:
        """Get top k indices by value"""
        if self.lib and self.tree_ptr:
            result_ptr = self.lib.query_top_k(self.tree_ptr, k, len(self.values))
            return [result_ptr[i] for i in range(min(k, len(self.values)))]
        else:
            # Fallback: sort indices by values
            indexed_values = [(val, idx) for idx, val in enumerate(self.values)]
            indexed_values.sort(reverse=True, key=lambda x: x[0])
            return [idx for _, idx in indexed_values[:k]]
    
    def query_range(self, left: int, right: int) -> float:
        """Get max value in range [left, right]"""
        if self.lib and self.tree_ptr:
            return self.lib.query_range_max(self.tree_ptr, left, right)
        else:
            if left > right or left < 0 or right >= len(self.values):
                return 0.0
            return max(self.values[left:right+1]) if self.values else 0.0
    
    def find_threshold_count(self, threshold: float) -> int:
        """Count how many values exceed threshold"""
        count = sum(1 for v in self.values if v >= threshold)
        return count
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile value"""
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * percentile / 100)
        return sorted_vals[idx] if sorted_vals else 0.0
    
    def destroy(self):
        if self.lib and self.tree_ptr:
            self.lib.destroy_segment_tree(self.tree_ptr)


class DoltDBManager:
    """Enhanced DoltDB manager with analytics"""
    
    def __init__(self, db_path="./ml_image_db"):
        self.db_path = Path(db_path)
        self.initialized = False
        self._initialize_db()
    
    def _initialize_db(self):
        console.print("[cyan]→ Initializing DoltDB...[/cyan]")
        
        if not self.db_path.exists():
            self.db_path.mkdir(parents=True)
            os.chdir(self.db_path)
            subprocess.run(["dolt", "init"], capture_output=True)
        else:
            os.chdir(self.db_path)
        
        self._create_tables()
        self.initialized = True
        console.print("[green]✓ DoltDB initialized[/green]")
    
    def _create_tables(self):
        # Test mode table (4 params)
        test_table = """
        CREATE TABLE IF NOT EXISTS testdb (
            id INT PRIMARY KEY AUTO_INCREMENT,
            learning_rate DOUBLE,
            kernel_size INT,
            epochs INT,
            batch_size INT,
            loss DOUBLE,
            psnr DOUBLE,
            ssim DOUBLE,
            mae DOUBLE,
            runtime DOUBLE,
            model_type VARCHAR(50) DEFAULT 'simple',
            timestamp DATETIME DEFAULT NOW()
        );
        """
        
        # Real mode table (8 params)
        real_table = """
        CREATE TABLE IF NOT EXISTS realdb (
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
            model_type VARCHAR(50) DEFAULT 'deep',
            timestamp DATETIME DEFAULT NOW()
        );
        """
        
        # Advanced mode table (U-Net)
        advanced_table = """
        CREATE TABLE IF NOT EXISTS advanceddb (
            id INT PRIMARY KEY AUTO_INCREMENT,
            learning_rate DOUBLE,
            kernel_size INT,
            epochs INT,
            batch_size INT,
            dropout_rate DOUBLE,
            weight_decay DOUBLE,
            loss DOUBLE,
            psnr DOUBLE,
            ssim DOUBLE,
            mae DOUBLE,
            runtime DOUBLE,
            model_type VARCHAR(50) DEFAULT 'unet',
            timestamp DATETIME DEFAULT NOW()
        );
        """
        
        self.execute_query(test_table, silent=True)
        self.execute_query(real_table, silent=True)
        self.execute_query(advanced_table, silent=True)
        
        subprocess.run(["dolt", "add", "."], capture_output=True)
        subprocess.run(["dolt", "commit", "-m", "Initial schema"], capture_output=True)
    
    def execute_query(self, query: str, description: str = "", silent: bool = False) -> Optional[str]:
        if not silent:
            console.print(f"[cyan]→ {description}[/cyan]")
        
        result = subprocess.run(
            ["dolt", "sql", "-q", query],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            if not silent:
                console.print(f"[red]✗ Query failed: {result.stderr}[/red]")
            return None
    
    def insert_result(self, mode: str, params: Dict, metrics: Dict):
        """Insert result into appropriate table"""
        if mode == "test":
            query = f"""
            INSERT INTO testdb (learning_rate, kernel_size, epochs, batch_size, 
                              loss, psnr, ssim, mae, runtime, model_type)
            VALUES ({params['lr']}, {params['kernel_size']}, {params['epochs']}, 
                    {params['batch_size']}, {metrics['loss']}, {metrics['psnr']}, 
                    {metrics['ssim']}, {metrics['mae']}, {metrics['runtime']}, 'simple');
            """
        elif mode == "real":
            query = f"""
            INSERT INTO realdb (learning_rate, momentum, kernel_size, stride, epochs, 
                              batch_size, dropout_rate, weight_decay, loss, psnr, 
                              ssim, mae, runtime, model_type)
            VALUES ({params['lr']}, {params['momentum']}, {params['kernel_size']}, 
                    {params['stride']}, {params['epochs']}, {params['batch_size']}, 
                    {params['dropout_rate']}, {params['weight_decay']}, {metrics['loss']}, 
                    {metrics['psnr']}, {metrics['ssim']}, {metrics['mae']}, 
                    {metrics['runtime']}, 'deep');
            """
        elif mode == "advanced":
            query = f"""
            INSERT INTO advanceddb (learning_rate, kernel_size, epochs, batch_size, 
                                  dropout_rate, weight_decay, loss, psnr, ssim, mae, 
                                  runtime, model_type)
            VALUES ({params['lr']}, {params['kernel_size']}, {params['epochs']}, 
                    {params['batch_size']}, {params['dropout_rate']}, 
                    {params['weight_decay']}, {metrics['loss']}, {metrics['psnr']}, 
                    {metrics['ssim']}, {metrics['mae']}, {metrics['runtime']}, 'unet');
            """
        
        self.execute_query(query, silent=True)
        subprocess.run(["dolt", "add", "."], capture_output=True)
        subprocess.run(["dolt", "commit", "-m", f"{mode} result: PSNR={metrics['psnr']:.2f}"], 
                      capture_output=True)
    
    def get_results(self, mode: str, limit: int = None) -> List[Dict]:
        """Get results from specified table"""
        table = f"{mode}db"
        query = f"SELECT * FROM {table} ORDER BY psnr DESC"
        if limit:
            query += f" LIMIT {limit};"
        else:
            query += ";"
        
        result = self.execute_query(query, silent=True)
        if result:
            return self._parse_results(result, mode)
        return []
    
    def _parse_results(self, output: str, mode: str) -> List[Dict]:
        """Parse query output into list of dicts"""
        lines = output.strip().split('\n')
        if len(lines) < 3:
            return []
        
        results = []
        for line in lines[3:]:
            if line.strip() and '|' in line:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 10:
                    try:
                        result = {
                            'id': int(parts[0]),
                            'lr': float(parts[1]),
                        }
                        
                        if mode == "test":
                            result.update({
                                'kernel_size': int(parts[2]),
                                'epochs': int(parts[3]),
                                'batch_size': int(parts[4]),
                                'loss': float(parts[5]),
                                'psnr': float(parts[6]),
                                'ssim': float(parts[7]),
                                'mae': float(parts[8]),
                                'runtime': float(parts[9])
                            })
                        elif mode == "real":
                            result.update({
                                'momentum': float(parts[2]),
                                'kernel_size': int(parts[3]),
                                'stride': int(parts[4]),
                                'epochs': int(parts[5]),
                                'batch_size': int(parts[6]),
                                'dropout_rate': float(parts[7]),
                                'weight_decay': float(parts[8]),
                                'loss': float(parts[9]),
                                'psnr': float(parts[10]),
                                'ssim': float(parts[11]),
                                'mae': float(parts[12]),
                                'runtime': float(parts[13])
                            })
                        elif mode == "advanced":
                            result.update({
                                'kernel_size': int(parts[2]),
                                'epochs': int(parts[3]),
                                'batch_size': int(parts[4]),
                                'dropout_rate': float(parts[5]),
                                'weight_decay': float(parts[6]),
                                'loss': float(parts[7]),
                                'psnr': float(parts[8]),
                                'ssim': float(parts[9]),
                                'mae': float(parts[10]),
                                'runtime': float(parts[11])
                            })
                        
                        results.append(result)
                    except:
                        pass
        return results
    
    def get_statistics(self, mode: str) -> Dict:
        """Get statistical summary"""
        query = f"""
        SELECT 
            COUNT(*) as count,
            AVG(psnr) as avg_psnr,
            MAX(psnr) as max_psnr,
            MIN(psnr) as min_psnr,
            AVG(ssim) as avg_ssim,
            AVG(loss) as avg_loss,
            AVG(runtime) as avg_runtime
        FROM {mode}db;
        """
        result = self.execute_query(query, silent=True)
        
        if result:
            lines = result.strip().split('\n')
            for line in lines[3:]:
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')[1:-1]]
                    if len(parts) >= 7:
                        try:
                            return {
                                'count': int(parts[0]),
                                'avg_psnr': float(parts[1]),
                                'max_psnr': float(parts[2]),
                                'min_psnr': float(parts[3]),
                                'avg_ssim': float(parts[4]),
                                'avg_loss': float(parts[5]),
                                'avg_runtime': float(parts[6])
                            }
                        except:
                            pass
        return {}


class HyperparameterGenerator:
    """Smart hyperparameter generation"""
    
    @staticmethod
    def generate_test_params(num: int = 10) -> List[Dict]:
        """Generate test mode parameters (4 params)"""
        params = []
        np.random.seed(42)
        
        for i in range(num):
            params.append({
                'lr': float(np.random.uniform(0.0001, 0.01)),
                'kernel_size': int(np.random.choice([3, 5, 7])),
                'epochs': int(np.random.randint(5, 15)),
                'batch_size': int(np.random.choice([16, 32, 64]))
            })
        
        return sorted(params, key=lambda x: x['lr'])
    
    @staticmethod
    def generate_real_params(num: int = 100) -> List[Dict]:
        """Generate real mode parameters (8 params)"""
        params = []
        np.random.seed(42)
        
        for i in range(num):
            quality_tier = i / num
            
            if quality_tier < 0.3:
                params.append({
                    'lr': float(np.random.uniform(0.0001, 0.005)),
                    'momentum': float(np.random.uniform(0.85, 0.95)),
                    'kernel_size': int(np.random.choice([3, 5, 7])),
                    'stride': int(np.random.choice([1, 2])),
                    'epochs': int(np.random.randint(8, 16)),
                    'batch_size': int(np.random.choice([32, 64])),
                    'dropout_rate': float(np.random.uniform(0.1, 0.3)),
                    'weight_decay': float(np.random.uniform(0.0001, 0.001))
                })
            elif quality_tier < 0.7:
                params.append({
                    'lr': float(np.random.uniform(0.005, 0.02)),
                    'momentum': float(np.random.uniform(0.7, 0.85)),
                    'kernel_size': int(np.random.choice([5, 7, 9])),
                    'stride': int(np.random.choice([1, 2, 3])),
                    'epochs': int(np.random.randint(4, 10)),
                    'batch_size': int(np.random.choice([16, 32])),
                    'dropout_rate': float(np.random.uniform(0.3, 0.5)),
                    'weight_decay': float(np.random.uniform(0.001, 0.01))
                })
            else:
                params.append({
                    'lr': float(np.random.uniform(0.02, 0.1)),
                    'momentum': float(np.random.uniform(0.5, 0.7)),
                    'kernel_size': int(np.random.choice([9, 11, 13])),
                    'stride': int(np.random.choice([2, 3, 4])),
                    'epochs': int(np.random.randint(2, 6)),
                    'batch_size': int(np.random.choice([8, 16])),
                    'dropout_rate': float(np.random.uniform(0.5, 0.7)),
                    'weight_decay': float(np.random.uniform(0.01, 0.1))
                })
        
        return sorted(params, key=lambda x: x['lr'])
    
    @staticmethod
    def generate_advanced_params(num: int = 50) -> List[Dict]:
        """Generate advanced mode parameters (U-Net)"""
        params = []
        np.random.seed(42)
        
        for i in range(num):
            params.append({
                'lr': float(np.random.uniform(0.0001, 0.01)),
                'kernel_size': int(np.random.choice([3, 5, 7])),
                'epochs': int(np.random.randint(5, 20)),
                'batch_size': int(np.random.choice([16, 32])),
                'dropout_rate': float(np.random.uniform(0.1, 0.4)),
                'weight_decay': float(np.random.uniform(0.0001, 0.01))
            })
        
        return sorted(params, key=lambda x: x['lr'])


class MLImageCLI:
    """Enhanced CLI with comprehensive features"""
    
    def __init__(self):
        self.db = None
        self.seg_tree = SegmentTreeWrapper()
        self.processor = None
        self.current_mode = None
        self.current_params = []
        self.current_results = []
        self.current_models = []
        self.output_dir = Path("./outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def initialize(self, image_path: str):
        console.print(Panel.fit("🚀 Enhanced ML Image Operation CLI", style="bold cyan"))
        
        self.processor = MLImageProcessor(image_path)
        self.db = DoltDBManager()
        
        # Display statistics
        for mode in ["test", "real", "advanced"]:
            stats = self.db.get_statistics(mode)
            if stats.get('count', 0) > 0:
                console.print(f"[green]  {mode.upper()}: {stats['count']} records, "
                            f"avg PSNR={stats['avg_psnr']:.2f}[/green]")
        
        console.print("[green]✓ Initialization complete[/green]\n")
    
    def select_mode(self) -> Optional[str]:
        console.print(Panel("Select Operating Mode", style="bold yellow"))
        console.print("[1] Test Mode - Quick (Simple Autoencoder, 4 params)")
        console.print("[2] Real Mode - Accurate (Deep Autoencoder, 8 params)")
        console.print("[3] Advanced Mode - Best Quality (U-Net, 6 params)")
        
        choice = console.input("\n[bold yellow]Select mode (1-3): [/bold yellow]")
        
        mode_map = {"1": "test", "2": "real", "3": "advanced"}
        return mode_map.get(choice)
    
    def run_training(self):
        """Run hyperparameter optimization"""
        mode = self.select_mode()
        if not mode:
            console.print("[red]Invalid choice[/red]")
            return
        
        # Get number of combinations
        if mode == "test":
            default_num = 10
        elif mode == "real":
            default_num = 100
        else:
            default_num = 50
        
        num = int(console.input(f"[cyan]Number of combinations (default {default_num}): [/cyan]") or str(default_num))
        
        self.current_mode = mode
        self.current_results = []
        self.current_models = []
        
        # Generate parameters
        if mode == "test":
            param_combinations = HyperparameterGenerator.generate_test_params(num)
        elif mode == "real":
            param_combinations = HyperparameterGenerator.generate_real_params(num)
        else:
            param_combinations = HyperparameterGenerator.generate_advanced_params(num)
        
        self.current_params = param_combinations
        self.seg_tree.create(len(param_combinations))
        
        console.print(Panel(f"Starting Training - {mode.upper()} Mode", style="bold yellow"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"[cyan]Training models...", total=len(param_combinations))
            
            for idx, params in enumerate(param_combinations):
                progress.update(task, description=f"[cyan]Combo {idx+1}/{len(param_combinations)}")
                
                # Train model
                if mode == "test":
                    model, metrics = self.processor.train_simple_model(params)
                elif mode == "real":
                    model, metrics = self.processor.train_deep_model(params)
                else:
                    model, metrics = self.processor.train_unet_model(params)
                
                self.current_results.append(metrics)
                self.current_models.append(model)
                
                # Update segment tree with PSNR
                self.seg_tree.update(idx, metrics['psnr'])
                
                # Save to database
                self.db.insert_result(mode, params, metrics)
                
                # Process and save image (every 10th or best so far)
                if idx % 10 == 0 or metrics['psnr'] == max(r['psnr'] for r in self.current_results):
                    output_path = self.output_dir / f"{mode}_result_{idx:04d}_psnr{metrics['psnr']:.1f}.png"
                    self.processor.process_image(model, str(output_path))
                
                progress.update(task, advance=1)
        
        console.print("[green]✓ Training complete![/green]")
        self._show_summary()
    
    def _show_summary(self):
        """Show training summary"""
        if not self.current_results:
            return
        
        psnr_values = [r['psnr'] for r in self.current_results]
        ssim_values = [r['ssim'] for r in self.current_results]
        
        table = Table(title="Training Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Combinations", str(len(self.current_results)))
        table.add_row("Best PSNR", f"{max(psnr_values):.2f} dB")
        table.add_row("Average PSNR", f"{np.mean(psnr_values):.2f} dB")
        table.add_row("Worst PSNR", f"{min(psnr_values):.2f} dB")
        table.add_row("Best SSIM", f"{max(ssim_values):.4f}")
        table.add_row("Average SSIM", f"{np.mean(ssim_values):.4f}")
        
        console.print(table)
    
    def query_top_k_tree(self):
        """Query top-k using segment tree"""
        if not self.current_params:
            console.print("[yellow]⚠ Run training first[/yellow]")
            return
        
        k = int(console.input("[cyan]Enter k (default 5): [/cyan]") or "5")
        indices = self.seg_tree.query_top_k(k)
        
        self._display_results(indices, f"Top-{k} Results (Segment Tree)")
    
    def query_top_k_db(self):
        """Query top-k from database"""
        mode = console.input("[cyan]Mode (test/real/advanced): [/cyan]").strip().lower()
        if mode not in ['test', 'real', 'advanced']:
            console.print("[red]Invalid mode[/red]")
            return
        
        k = int(console.input("[cyan]Enter k (default 5): [/cyan]") or "5")
        results = self.db.get_results(mode, limit=k)
        
        self._display_db_results(results, f"Top-{k} from {mode.upper()}")
    
    def query_range(self):
        """Query range using segment tree"""
        if not self.current_params:
            console.print("[yellow]⚠ Run training first[/yellow]")
            return
        
        min_lr = float(console.input("[cyan]Min learning rate: [/cyan]"))
        max_lr = float(console.input("[cyan]Max learning rate: [/cyan]"))
        
        lr_list = [p['lr'] for p in self.current_params]
        left = bisect.bisect_left(lr_list, min_lr)
        right = bisect.bisect_right(lr_list, max_lr) - 1
        
        if left > right:
            console.print("[yellow]⚠ No results in range[/yellow]")
            return
        
        max_psnr = self.seg_tree.query_range(left, right)
        console.print(f"[green]Max PSNR in LR [{min_lr:.4f}, {max_lr:.4f}]: {max_psnr:.2f} dB[/green]")
        
        # Find best configuration in range
        best_idx = left
        for i in range(left, right + 1):
            if self.current_results[i]['psnr'] >= max_psnr:
                best_idx = i
                break
        
        console.print(f"\n[cyan]Best configuration in range:[/cyan]")
        self._display_single_result(best_idx)
    
    def query_threshold(self):
        """Find all configs above PSNR threshold"""
        if not self.current_results:
            console.print("[yellow]⚠ Run training first[/yellow]")
            return
        
        threshold = float(console.input("[cyan]PSNR threshold: [/cyan]"))
        count = self.seg_tree.find_threshold_count(threshold)
        
        console.print(f"[green]Found {count} configurations with PSNR ≥ {threshold:.2f} dB[/green]")
        
        # Show top configurations above threshold
        above_threshold = [(idx, r['psnr']) for idx, r in enumerate(self.current_results) if r['psnr'] >= threshold]
        above_threshold.sort(key=lambda x: x[1], reverse=True)
        
        if above_threshold:
            console.print(f"\n[cyan]Top 5 above threshold:[/cyan]")
            self._display_results([idx for idx, _ in above_threshold[:5]], "Above Threshold")
    
    def analyze_percentiles(self):
        """Analyze performance percentiles"""
        if not self.current_results:
            console.print("[yellow]⚠ Run training first[/yellow]")
            return
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        
        table = Table(title="Performance Percentiles", box=box.ROUNDED)
        table.add_column("Percentile", style="cyan")
        table.add_column("PSNR (dB)", style="green")
        table.add_column("SSIM", style="magenta")
        
        for p in percentiles:
            psnr_val = self.seg_tree.get_percentile(p)
            ssim_vals = sorted([r['ssim'] for r in self.current_results])
            ssim_val = ssim_vals[int(len(ssim_vals) * p / 100)]
            table.add_row(f"{p}th", f"{psnr_val:.2f}", f"{ssim_val:.4f}")
        
        console.print(table)
    
    def compare_modes(self):
        """Compare performance across modes"""
        modes = ['test', 'real', 'advanced']
        
        table = Table(title="Mode Comparison", box=box.ROUNDED)
        table.add_column("Mode", style="cyan")
        table.add_column("Count", style="yellow")
        table.add_column("Avg PSNR", style="green")
        table.add_column("Max PSNR", style="magenta")
        table.add_column("Avg Runtime", style="blue")
        
        for mode in modes:
            stats = self.db.get_statistics(mode)
            if stats.get('count', 0) > 0:
                table.add_row(
                    mode.upper(),
                    str(stats['count']),
                    f"{stats['avg_psnr']:.2f}",
                    f"{stats['max_psnr']:.2f}",
                    f"{stats['avg_runtime']:.1f}s"
                )
        
        console.print(table)
    
    def visualize_results(self):
        """Create visualizations"""
        if not self.current_results:
            console.print("[yellow]⚠ Run training first[/yellow]")
            return
        
        psnr_values = [r['psnr'] for r in self.current_results]
        ssim_values = [r['ssim'] for r in self.current_results]
        lr_values = [p['lr'] for p in self.current_params]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PSNR distribution
        axes[0, 0].hist(psnr_values, bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('PSNR Distribution')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_ylabel('Frequency')
        
        # SSIM distribution
        axes[0, 1].hist(ssim_values, bins=20, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('SSIM Distribution')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_ylabel('Frequency')
        
        # Learning Rate vs PSNR
        axes[1, 0].scatter(lr_values, psnr_values, alpha=0.6)
        axes[1, 0].set_title('Learning Rate vs PSNR')
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_xscale('log')
        
        # PSNR vs SSIM
        axes[1, 1].scatter(psnr_values, ssim_values, alpha=0.6, color='green')
        axes[1, 1].set_title('PSNR vs SSIM')
        axes[1, 1].set_xlabel('PSNR (dB)')
        axes[1, 1].set_ylabel('SSIM')
        
        plt.tight_layout()
        output_path = self.output_dir / f"analysis_{self.current_mode}_{int(time.time())}.png"
        plt.savefig(output_path, dpi=150)
        console.print(f"[green]✓ Visualization saved: {output_path}[/green]")
        plt.close()
    
    def export_best_model(self):
        """Export the best model"""
        if not self.current_models or not self.current_results:
            console.print("[yellow]⚠ Run training first[/yellow]")
            return
        
        best_idx = max(range(len(self.current_results)), 
                      key=lambda i: self.current_results[i]['psnr'])
        
        model_path = self.output_dir / f"best_model_{self.current_mode}.pth"
        import torch
        torch.save(self.current_models[best_idx].state_dict(), model_path)
        
        console.print(f"[green]✓ Best model saved: {model_path}[/green]")
        console.print(f"[cyan]  PSNR: {self.current_results[best_idx]['psnr']:.2f} dB[/cyan]")
        console.print(f"[cyan]  Parameters: {self.current_params[best_idx]}[/cyan]")
    
    def custom_sql(self):
        """Execute custom SQL query"""
        query = console.input("[cyan]Enter SQL query: [/cyan]")
        result = self.db.execute_query(query, "Custom Query")
        if result:
            console.print(Panel(result, title="Query Result", border_style="green"))
    
    def _display_results(self, indices: List[int], title: str):
        """Display results from current session"""
        table = Table(title=title, box=box.ROUNDED, show_header=True)
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("LR", style="green")
        
        if self.current_mode == "real":
            table.add_column("Momentum", style="green")
        
        table.add_column("Kernel", style="green")
        
        if self.current_mode == "real":
            table.add_column("Stride", style="green")
        
        table.add_column("Epochs", style="green")
        table.add_column("PSNR", style="magenta")
        table.add_column("SSIM", style="blue")
        table.add_column("MAE", style="yellow")
        
        for rank, idx in enumerate(indices, 1):
            if idx < len(self.current_params):
                params = self.current_params[idx]
                metrics = self.current_results[idx]
                
                row = [str(rank), f"{params['lr']:.4f}"]
                
                if self.current_mode == "real":
                    row.append(f"{params['momentum']:.3f}")
                
                row.append(str(params['kernel_size']))
                
                if self.current_mode == "real":
                    row.append(str(params['stride']))
                
                row.extend([
                    str(params['epochs']),
                    f"{metrics['psnr']:.2f}",
                    f"{metrics['ssim']:.4f}",
                    f"{metrics['mae']:.2f}"
                ])
                
                table.add_row(*row)
        
        console.print(table)
    
    def _display_db_results(self, results: List[Dict], title: str):
        """Display results from database"""
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Rank", style="cyan")
        table.add_column("LR", style="green")
        table.add_column("Kernel", style="green")
        table.add_column("Epochs", style="green")
        table.add_column("PSNR", style="magenta")
        table.add_column("SSIM", style="blue")
        table.add_column("MAE", style="yellow")
        
        for rank, result in enumerate(results, 1):
            table.add_row(
                str(rank),
                f"{result['lr']:.4f}",
                str(result['kernel_size']),
                str(result['epochs']),
                f"{result['psnr']:.2f}",
                f"{result['ssim']:.4f}",
                f"{result.get('mae', 0):.2f}"
            )
        
        console.print(table)
    
    def _display_single_result(self, idx: int):
        """Display single result"""
        params = self.current_params[idx]
        metrics = self.current_results[idx]
        
        tree = Tree(f"[bold cyan]Configuration #{idx}[/bold cyan]")
        
        params_node = tree.add("[yellow]Parameters[/yellow]")
        for key, val in params.items():
            params_node.add(f"{key}: {val}")
        
        metrics_node = tree.add("[green]Metrics[/green]")
        for key, val in metrics.items():
            if isinstance(val, float):
                metrics_node.add(f"{key}: {val:.4f}")
            else:
                metrics_node.add(f"{key}: {val}")
        
        console.print(tree)
    
    def show_menu(self):
        """Main interactive menu"""
        while True:
            console.print("\n" + "="*70)
            console.print(Panel.fit("ML Image CLI - Main Menu", style="bold cyan"))
            console.print("="*70)
            
            menu = """
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
            """
            console.print(menu)
            
            choice = console.input("[bold yellow]Select option (1-12): [/bold yellow]")
            
            try:
                if choice == "1":
                    self.run_training()
                elif choice == "2":
                    self.export_best_model()
                elif choice == "3":
                    self.query_top_k_tree()
                elif choice == "4":
                    self.query_range()
                elif choice == "5":
                    self.query_threshold()
                elif choice == "6":
                    self.analyze_percentiles()
                elif choice == "7":
                    self.query_top_k_db()
                elif choice == "8":
                    self.compare_modes()
                elif choice == "9":
                    self.custom_sql()
                elif choice == "10":
                    self.visualize_results()
                elif choice == "11":
                    self.show_statistics()
                elif choice == "12":
                    console.print("[yellow]Goodbye![/yellow]")
                    self.seg_tree.destroy()
                    break
                else:
                    console.print("[red]Invalid option[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation cancelled[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def show_statistics(self):
        """Show comprehensive statistics"""
        console.print(Panel("System Statistics", style="bold cyan"))
        
        for mode in ["test", "real", "advanced"]:
            stats = self.db.get_statistics(mode)
            if stats.get('count', 0) > 0:
                table = Table(title=f"{mode.upper()} Mode Statistics", box=box.SIMPLE)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Total Configurations", str(stats['count']))
                table.add_row("Average PSNR", f"{stats['avg_psnr']:.2f} dB")
                table.add_row("Max PSNR", f"{stats['max_psnr']:.2f} dB")
                table.add_row("Min PSNR", f"{stats['min_psnr']:.2f} dB")
                table.add_row("Average SSIM", f"{stats['avg_ssim']:.4f}")
                table.add_row("Average Loss", f"{stats['avg_loss']:.4f}")
                table.add_row("Average Runtime", f"{stats['avg_runtime']:.1f}s")
                
                console.print(table)
                console.print()


@click.command()
@click.option('--image', '-i', default='sample_image.png', help='Input image path')
def main(image):
    """Enhanced ML Image Operation CLI - Main Entry Point"""
    try:
        if not Path(image).exists():
            console.print(f"[red]Error: Image file not found: {image}[/red]")
            console.print("[yellow]Please provide a valid image path using --image option[/yellow]")
            sys.exit(1)
        
        cli = MLImageCLI()
        cli.initialize(image)
        cli.show_menu()
    except KeyboardInterrupt:
        console.print("\n[yellow]Application terminated by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
