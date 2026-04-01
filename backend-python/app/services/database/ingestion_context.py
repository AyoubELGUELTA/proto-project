import logging

class IngestionContext:
    """
    Assure la cohérence du statut du document dans Postgres.
    Garantit que le document ne reste pas bloqué en 'PROCESSING' si le code crash.
    """
    def __init__(self, doc_repo, doc_id: str):
        self.doc_repo = doc_repo
        self.doc_id = doc_id

    async def __aenter__(self):
        """Action automatique lors du 'async with'."""
        await self.doc_repo.set_status_processing(self.doc_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Action automatique à la sortie du bloc (succès ou erreur)."""
        if exc_type is not None:
            logging.error(f"❌ Erreur critique sur le doc {self.doc_id}: {exc_val}")
            await self.doc_repo.set_status_failed(self.doc_id)
        else:
            await self.doc_repo.set_status_completed(self.doc_id)