from app.models.restricted_areas import RestrictedArea
from app.repositories.event_repository_base import EventRepositoryBase


class RestrictedAreaRepository(EventRepositoryBase):
    model = RestrictedArea
