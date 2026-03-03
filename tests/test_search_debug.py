import os
import asyncio
from config.settings import load_config
from main import setup_app

def test():
    print("Loading config...")
    cfg = load_config()
    print("Initializing app...")
    app = setup_app()
    print("Testing search...")
    print(f"Internet Enabled: {cfg.internet_enabled}")
    print(f"Search API: {cfg.search_api_url}")
    print(f"Safety Mode: {cfg.safety_mode}")
    
    # We test the web stage manually, outside the console
    res = app.brain.handle_message("/web курс доллара")
    print("\n--- RESULTS ---")
    print(res.text)

if __name__ == "__main__":
    test()
