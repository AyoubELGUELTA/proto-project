import logging
import sys

def setup_logging():
    """
    Centralized logging configuration for production and development.
    Silences noisy libraries and 
    ensuring application logs are visible in the terminal.
    """
    # 1. Format identique à ton fichier pytest.ini
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%H:%M:%S"
    
    # 2. Configuration du handler racine (Stream to stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Évite de dupliquer les handlers si setup_logging est appelé plusieurs fois
    if not root_logger.handlers:
        root_logger.addHandler(handler)
        
    # 3. SILENCE LE BRUIT - Mapping identique à ton pytest.ini
    noisy_libraries = {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "openai": logging.INFO,
        "urllib3": logging.WARNING,
        "neo4j": logging.WARNING,       # On fait taire les avertissements DBMS de Neo4j
        "asyncio": logging.WARNING
    }
    
    for lib_name, log_level in noisy_libraries.items():
        logging.getLogger(lib_name).setLevel(log_level)
        
    logging.getLogger("uvicorn.error").info("🛠️ Centralized logging system successfully initialized.")