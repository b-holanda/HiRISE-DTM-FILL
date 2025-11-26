import sys
from pathlib import Path

# Garante que o pacote local "marsfill" seja importável durante os testes
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
