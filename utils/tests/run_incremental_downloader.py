from core.downloader import download_price_data

symbols = ["RELIANCE", "TCS", "INFY"]

for s in symbols:
    print("Downloading:", s)
    try:
        ok = download_price_data(s, force_refresh=False)
        print(s, "->", 'OK' if ok else 'FAILED/NO_DATA')
    except Exception as e:
        print(s, "-> EXCEPTION:", e)
