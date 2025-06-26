import joblib
import logging
import signal
import sys
from src.bot import run_trading_bot

# --- Config: Set your model paths here ---
CLASSIFIER_PATH = "models/best_xgb_classifier.pkl"
REGRESSOR_PATH = "models/best_xgb_regressor.pkl"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, filename="bot.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

def load_models():
    logging.info("Loading models...")
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    logging.info("Models loaded successfully")
    return clf, reg

def signal_handler(sig, frame):
    logging.info("Shutdown signal received. Exiting gracefully...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    classifier, regressor = load_models()

    import src.alpaca_api as alpaca_api
    actions = alpaca_api.User_Actions()
    actions.is_market_open = lambda: True

    logging.info("Starting trading bot in PAPER TRADING mode.")
    try:
        run_trading_bot(classifier, regressor,
                        clf_threshold=0.58,
                        reg_threshold=0.002)
    except Exception as e:
        logging.error(f"Exception in bot run: {e}", exc_info=True)


if __name__ == "__main__":
    main()
