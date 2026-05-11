import logging

logger = logging.getLogger(__name__)

class IngestionContext:
    """
    Manages document ingestion status within PostgreSQL using the Context Manager pattern.
    
    Ensures that document status is updated to 'PROCESSING' upon entry and 
    automatically transitions to either 'COMPLETED' or 'FAILED' upon exit, 
    even in the event of an unhandled exception.
    """
    
    def __init__(self, doc_repo, doc_id: str):
        """
        Initializes the context manager.

        Args:
            doc_repo (DocumentRepository): The repository handling SQL updates.
            doc_id (str): The unique identifier of the document being processed.
        """
        self.doc_repo = doc_repo
        self.doc_id = doc_id

    async def __aenter__(self):
        """
        Triggered when entering the 'async with' block.
        Marks the document as currently being processed.
        """
        logger.info(f"🚦 Document {self.doc_id} entering PROCESSING state.")
        await self.doc_repo.set_status_processing(self.doc_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Triggered when exiting the 'async with' block.
        
        Args:
            exc_type: The exception class (if any).
            exc_val: The exception instance (if any).
            exc_tb: The traceback (if any).
        """
        if exc_type is not None:
            # An error occurred during the workflow
            logger.error(f"❌ Critical failure during ingestion of doc {self.doc_id}: {exc_val}")
            await self.doc_repo.set_status_failed(self.doc_id)
            # We don't suppress the exception; it will continue to propagate
        else:
            # Workflow finished successfully
            logger.info(f"✅ Ingestion successful. Marking doc {self.doc_id} as COMPLETED.")
            await self.doc_repo.set_status_completed(self.doc_id)