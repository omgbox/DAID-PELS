"""
BookBot Training Visualizer
Live training status and progress visualization.
"""

import time
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger('bookbot.training.visualizer')


class TrainingVisualizer:
    """Live training visualization and status."""

    def __init__(self):
        self.start_time = None
        self.current_pass = 0
        self.total_passes = 0
        self.current_step = ""
        self.stats = {}
        self.history = []

    def start(self, total_passes: int):
        """Start training visualization."""
        self.start_time = time.time()
        self.total_passes = total_passes
        self.current_pass = 0
        self.print_header()

    def print_header(self):
        """Print training header."""
        print("\n" + "=" * 70)
        print("  BOOKBOT TRAINING")
        print("=" * 70)
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Passes: {self.total_passes}")
        print("=" * 70 + "\n")

    def start_pass(self, pass_num: int, pass_name: str):
        """Start a new pass."""
        self.current_pass = pass_num
        self.current_step = pass_name
        elapsed = self._get_elapsed()
        print(f"\n[{elapsed}] PASS {pass_num}: {pass_name}")
        print("-" * 50)

    def update_step(self, step: str, progress: float = None):
        """Update current step."""
        self.current_step = step
        elapsed = self._get_elapsed()
        if progress is not None:
            bar = self._progress_bar(progress)
            print(f"  {elapsed} {step} {bar}", end='\r')
        else:
            print(f"  {elapsed} {step}")

    def update_stats(self, stats: Dict):
        """Update statistics."""
        self.stats.update(stats)

    def print_stats(self):
        """Print current statistics."""
        elapsed = self._get_elapsed()
        print(f"\n  [{elapsed}] Statistics:")
        for key, value in self.stats.items():
            if isinstance(value, int) and value > 0:
                print(f"    {key}: {value:,}")
            elif isinstance(value, float):
                print(f"    {key}: {value:.4f}")

    def end_pass(self, pass_num: int, duration: float):
        """End a pass."""
        elapsed = self._get_elapsed()
        print(f"\n  [{elapsed}] Pass {pass_num} completed in {duration:.1f}s")
        self.history.append({
            'pass': pass_num,
            'duration': duration,
            'stats': dict(self.stats),
        })

    def end(self):
        """End training."""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Total time: {self._format_time(total_time)}")
        print(f"  Passes completed: {len(self.history)}")
        print("\n  Pass Summary:")
        for h in self.history:
            print(f"    Pass {h['pass']}: {h['duration']:.1f}s")
        print("=" * 70 + "\n")

    def _get_elapsed(self) -> str:
        """Get elapsed time string."""
        if not self.start_time:
            return "00:00"
        elapsed = time.time() - self.start_time
        return self._format_time(elapsed)

    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a progress bar."""
        filled = int(width * progress)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {progress*100:.1f}%"


class BatchProcessor:
    """Process items in batches with progress tracking."""

    def __init__(self, items: List, batch_size: int = 100, visualizer: TrainingVisualizer = None):
        self.items = items
        self.batch_size = batch_size
        self.visualizer = visualizer
        self.processed = 0
        self.total = len(items)

    def process(self, func, *args, **kwargs) -> List:
        """
        Process items in batches.

        Args:
            func: Function to apply to each item

        Returns:
            List of results
        """
        results = []
        for i in range(0, self.total, self.batch_size):
            batch = self.items[i:i + self.batch_size]
            for item in batch:
                result = func(item, *args, **kwargs)
                results.append(result)
                self.processed += 1

            # Update progress
            if self.visualizer:
                progress = self.processed / self.total
                self.visualizer.update_step(
                    f"Processed {self.processed}/{self.total}",
                    progress
                )

        return results


def create_progress_callback(visualizer: TrainingVisualizer):
    """Create a progress callback function."""
    def callback(step: str, current: int, total: int):
        progress = current / total if total > 0 else 0
        visualizer.update_step(step, progress)
    return callback
