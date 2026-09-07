from fastapi import APIRouter

from app.ai.tts.voices import VOICE_CATEGORIES

router = APIRouter(tags=["voices"])


@router.get("/voices")
async def list_voices():
    # Return Voice Design categories and attributes. No presets - user selects per category.
    categories = []
    for cat_id, cat in VOICE_CATEGORIES.items():
        categories.append(
            {
                "id": cat_id,
                "label": cat["label"],
                "attributes": cat["attributes"],
            }
        )
    return {"categories": categories}
