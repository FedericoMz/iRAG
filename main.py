from irag.api import app
from tools.logger import logger


if __name__ == "__main__":
    import uvicorn

    logger.info("App starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
