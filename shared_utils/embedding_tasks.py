import time, logging

from celery import shared_task

from collections import deque
from django.db import transaction
from shared_utils.entities.service.embedding_model import _embed_sync
from shared_utils.entities.EntityModel import EntityModel


_BATCH_SIZE    = 64          # tune to GPU / RAM
_MAX_WAIT_SEC  = 1      # flush every 1 sec at worst

_buffer   = deque()          # (id, text)
_deadline = None

logger = logging.getLogger(__name__)

@shared_task(bind=True, acks_late=True, queue="embedding")
def enqueue_embedding_task(self, entity_id: str, raw_text: str):
    """
    Called once per entity save.  We buffer and let `flush_batch`
    decide when to actually run the embed & bulk_update.
    """
    logger.info(f"Enqueueing embedding task for entity {entity_id}")
    global _deadline
    _buffer.append((entity_id, raw_text))

    if _deadline is None:                      # first item → start timer
        _deadline = time.monotonic() + _MAX_WAIT_SEC

    if len(_buffer) >= _BATCH_SIZE or time.monotonic() >= _deadline:
        flush_batch()


def flush_batch():
    global _buffer, _deadline
    if not _buffer:
        return

    ids, texts = zip(*_buffer)
    logger.debug("Embedding batch of %d", len(ids))

    vectors = _embed_sync(list(texts))         # ONE forward-pass

    # bulk_update in one transaction
    with transaction.atomic():
        objs = list(EntityModel.objects.filter(pk__in=ids))
        for obj, vec in zip(objs, vectors):
            obj.embedding = vec               # 1-D list ➜ pgvector OK
        EntityModel.objects.bulk_update(objs, ["embedding"])

    _buffer.clear()
    _deadline = None
