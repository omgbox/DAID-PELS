"""
DAID-PELS Server Entry Point
Run with: python -m server
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .app import create_app


def main():
    """Run the server."""
    app = create_app()
    
    print("\n" + "=" * 60)
    print("  DAID-PELS Server")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
