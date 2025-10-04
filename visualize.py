# SegMaster/visualize.py

from rich.console import Console

console = Console()

def visualize_layout(layout):
    s = ""
    for val in layout:
        if val == 1:
            s += "[red]█[/red]"
        else:
            s += "[green]░[/green]"
    return s
