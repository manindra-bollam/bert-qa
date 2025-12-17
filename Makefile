start:
	uvicorn app:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app:app --reload
