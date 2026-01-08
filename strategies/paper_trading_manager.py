# Artemis Signals — Paper Trading Manager (Stub)
# For Shivaansh & Krishaansh — this file pays your fees!

class PaperTradingManager:
    def __init__(self):
        self.active = False
    def start(self):
        self.active = True
    def stop(self):
        self.active = False
    def status(self):
        return "Paper trading active" if self.active else "Paper trading inactive"
