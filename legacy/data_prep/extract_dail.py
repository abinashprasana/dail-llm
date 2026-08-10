"""
Shim: the real implementation lives at dail_llm/data/extract_dail.py
"""
from dail_llm.data.extract_dail import extract  # noqa: F401

if __name__ == "__main__":
    extract()
