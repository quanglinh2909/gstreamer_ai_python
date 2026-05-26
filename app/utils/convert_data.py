from app.enum.config_ai_enum import TypeConfigAiEnum


def convert_config_type(value: str):
    if value == TypeConfigAiEnum.PLATE_RECOGNITION.value:
        return "align_plate"
    elif value == TypeConfigAiEnum.FACE_RECOGNITION.value:
        return "align_face"
    else:
        return value
