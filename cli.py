# SegMaster/cli.py

import subprocess
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import IntPrompt

console = Console()
history = []

def add_history(msg):
    history.append(msg)
    if len(history) > 5:
        history.pop(0)

class MemorySimulator:
    def __init__(self):
        self.p = None
        self.n = None
        self.start_process()

    def start_process(self):
        self.p = subprocess.Popen(["./build/segment"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    def send_cmd(self, cmd):
        start = time.time()
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        response = self.p.stdout.readline().strip()
        end = time.time()
        timing = (end - start) * 1000
        return response, timing

    def init_memory(self, n):
        resp, _ = self.send_cmd(f"init {n}")
        if resp == "OK":
            self.n = n
            return True
        return False

    def allocate(self, k):
        resp, timing = self.send_cmd(f"alloc {k}")
        if resp == "FAIL":
            return None, timing
        else:
            return int(resp), timing

    def free(self, L, R):
        resp, timing = self.send_cmd(f"free {L} {R}")
        return resp == "OK", timing

    def update_range(self, L, R, val):
        resp, timing = self.send_cmd(f"update {L} {R} {val}")
        return resp == "OK", timing

    def total_allocated(self):
        resp, _ = self.send_cmd("total_alloc")
        return int(resp)

    def max_free_block(self):
        resp, _ = self.send_cmd("max_free")
        return int(resp)

    def find_first_fit(self, k):
        resp, _ = self.send_cmd(f"find_first {k}")
        if resp == "NONE":
            return None
        return int(resp)

    def reset(self):
        resp, _ = self.send_cmd("reset")
        return resp == "OK"

    def get_layout(self):
        resp, _ = self.send_cmd("dump")
        return list(map(int, resp.split()))

    def close(self):
        self.send_cmd("exit")
        self.p.terminate()

def show_menu():
    console.print(Panel("🧠 Memory Allocator Simulator", style="bold blue"))
    table = Table(show_header=False, show_lines=True)
    table.add_row("1", "Initialize memory pool of size N")
    table.add_row("2", "Allocate memory block of size k (first-fit)")
    table.add_row("3", "Free memory block range [L,R]")
    table.add_row("4", "Find first free memory block of size ≥ k")
    table.add_row("5", "Show memory usage (allocated / free / fragmentation)")
    table.add_row("6", "Display memory layout (ASCII + colored blocks)")
    table.add_row("7", "Update memory range (allocate or free using lazy propagation)")
    table.add_row("8", "Show largest contiguous free block")
    table.add_row("9", "Reset memory pool")
    table.add_row("10", "Show operation history")
    table.add_row("11", "Exit")
    console.print(table)

def main():
    sim = MemorySimulator()
    while True:
        show_menu()
        choice = IntPrompt.ask("Select option")
        if choice == 1:
            if sim.n is not None:
                console.print("[yellow]Memory already initialized. Reset first?[/yellow]")
                continue
            n = IntPrompt.ask("Enter memory size N")
            if sim.init_memory(n):
                add_history(f"Initialized memory of size {n}")
                console.print("[green]✅ Initialized successfully[/green]")
            else:
                console.print("[red]❌ Failed to initialize[/red]")
        elif choice == 2:
            if sim.n is None:
                console.print("[yellow]⚠️ Initialize memory first[/yellow]")
                continue
            k = IntPrompt.ask("Enter size k")
            pos, timing = sim.allocate(k)
            if pos is None:
                add_history(f"Allocate {k} failed")
                console.print("[red]❌ Allocation failed[/red]")
            else:
                add_history(f"Allocated {k} at {pos}")
                console.print(f"[green]✅ Allocated at position {pos}[/green]")
            console.print(f"Time: {timing:.2f} ms")
        elif choice == 3:
            if sim.n is None:
                console.print("[yellow]⚠️ Initialize memory first[/yellow]")
                continue
            L = IntPrompt.ask("Enter L")
            R = IntPrompt.ask("Enter R")
            success, timing = sim.free(L, R)
            if success:
                add_history(f"Freed [{L},{R}]")
                console.print("[green]✅ Freed successfully[/green]")
            else:
                console.print("[red]❌ Free failed[/red]")
            console.print(f"Time: {timing:.2f} ms")
        elif choice == 4:
            if sim.n is None:
                console.print("[yellow]⚠️ Initialize memory first[/yellow]")
                continue
            k = IntPrompt.ask("Enter k")
            pos = sim.find_first_fit(k)
            if pos is None:
                console.print("[yellow]⚠️ No such block[/yellow]")
            else:
                console.print(f"[green]First fit at {pos}[/green]")
        elif choice == 5:
            if sim.n is None:
                console.print("[yellow]⚠️ Initialize memory first[/yellow]")
                continue
            alloc = sim.total_allocated()
            free_amt = sim.n - alloc
            maxf = sim.max_free_block()
            frag = 0 if free_amt == 0 else (1 - maxf / free_amt) * 100
            console.print(f"Allocated: {alloc}")
            console.print(f"Free: {free_amt}")
            console.print(f"Fragmentation: {frag:.2f}%")
        elif choice == 6:
            if sim.n is None:
                console.print("[yellow]⚠️ Initialize memory first[/yellow]")
                continue
            layout = sim.get_layout()
            from visualize import visualize_layout  # Import here if separate file
            vis = visualize_layout(layout)
            console.print(Panel(vis, title="Memory Layout"))
        elif choice == 7:
            if sim.n is None:
                console.print("[yellow]⚠️ Initialize memory first[/yellow]")
                continue
            L = IntPrompt.ask("Enter L")
            R = IntPrompt.ask("Enter R")
            val = IntPrompt.ask("Enter value (0 free, 1 alloc)", choices=[0, 1])
            success, timing = sim.update_range(L, R, val)
            if success:
                add_history(f"Updated [{L},{R}] to {val}")
                console.print("[green]✅ Updated[/green]")
            else:
                console.print("[red]❌ Update failed[/red]")
            console.print(f"Time: {timing:.2f} ms")
        elif choice == 8:
            if sim.n is None:
                console.print("[yellow]⚠️ Initialize memory first[/yellow]")
                continue
            maxf = sim.max_free_block()
            console.print(f"Largest free block: {maxf}")
        elif choice == 9:
            if sim.n is None:
                console.print("[yellow]⚠️ Initialize memory first[/yellow]")
                continue
            success = sim.reset()
            if success:
                add_history("Reset memory")
                console.print("[green]✅ Reset successful[/green]")
        elif choice == 10:
            console.print(Panel("\n".join(history), title="Last 5 Operations"))
        elif choice == 11:
            sim.close()
            break
        else:
            console.print("[red]Invalid choice[/red]")

if __name__ == "__main__":
    main()
