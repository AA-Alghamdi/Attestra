"""Allow running attestra as a module: python -m attestra run --goal ..."""
import sys
from .cli.main import main
sys.exit(main())
