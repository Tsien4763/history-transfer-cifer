from pathlib import Path
import tempfile
ROOT = Path(__file__).resolve().parents[1]
R = ROOT
OUT = Path(tempfile.gettempdir()) / 'history_transfer_mvp_out'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'cache'
