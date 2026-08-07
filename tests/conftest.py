import sys
import os

# Add src/ to path so tests can import project modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# agents.yaml is now Ollama-only and interpolates host: "${OLLAMA_HOST}", which
# load_config requires. Provide a default so config-loading tests don't need a
# real Ollama server; tests that assert the unset-error delete it explicitly.
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
