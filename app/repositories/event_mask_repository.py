from app.models.event_mask import EventMask
from app.repositories.event_repository_base import EventRepositoryBase


class EventMaskRepository(EventRepositoryBase):
    model = EventMask
