from app.models.event_plate import EventPlate
from app.repositories.event_repository_base import EventRepositoryBase


class EventPlateRepository(EventRepositoryBase):
    model = EventPlate
