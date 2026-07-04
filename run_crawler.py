"""Run crawler multiple times."""
import subprocess

for i in range(5):
    print(f"\n{'='*60}")
    print(f"  Run {i+1}/5")
    print(f"{'='*60}")
    result = subprocess.run(['python', 'wiki_crawler.py'], capture_output=True, text=True)
    # Print last few lines
    lines = result.stdout.strip().split('\n')
    for line in lines[-5:]:
        print(line)
