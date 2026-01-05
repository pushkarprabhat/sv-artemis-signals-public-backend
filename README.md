# sv-artemis-signals-public-backend

**Clean commercial backend for Artemis Signals by ManekBaba**  
Institutional-grade quant trading API — pairs, IV Crush, GARCH, Kelly sizing, backtesting.  

**Features**:
- FastAPI endpoints
- Clean, no personal/family data
- Ready for subscription gating


**Setup**:
- Install: `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and fill in all required values for email/SMS/WhatsApp alerts and admin contacts.
- Always run scripts and the server from this directory (`sv-artemis-signals-public-backend`).
- Set your PYTHONPATH to include this directory for all scripts:
	- On Linux/macOS: `export PYTHONPATH=$(pwd):$PYTHONPATH`
	- On Windows (PowerShell): `$env:PYTHONPATH=(Get-Location).Path + ';' + $env:PYTHONPATH`
- Run: `uvicorn api.server:app --reload`
- Docs: Swagger at /docs

**Troubleshooting**:
- If you see config import errors, check that you are running from the correct directory and that PYTHONPATH is set as above.
- All notification settings are config-driven. See `.env.example` for required variables.

**Designed & Developed by TheiaOne Ventures**
