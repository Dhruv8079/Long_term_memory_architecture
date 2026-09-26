# Contributing

1. Fork and create a feature branch.
2. Copy `.env.example` to `.env` (never commit `.env`).
3. Run `psql $DATABASE_URL -f schema.sql` once.
4. Install dev deps: `pip install -r requirements-dev.txt`.
5. Verify: `python -m py_compile buffer.py long_term_memory.py config.py database.py test_neon.py` and `python test_neon.py`.
6. Open a PR with a clear description and test notes.
