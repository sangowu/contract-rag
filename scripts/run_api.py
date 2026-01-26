import uvicorn
import os
import sys
from loguru import logger

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logger.info(f"Project root: {project_root}")
    uvicorn.run(
        "api.rag_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )

if __name__ == "__main__":
    main()