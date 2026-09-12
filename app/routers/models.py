from fastapi import APIRouter, HTTPException

from app.ai.llm import list_models

router = APIRouter(tags=['models'])


@router.get('/models')
async def get_models():
    """
    Return the list of model names served by the llama.cpp server.
    Useful for populating a model selector in the client UI.
    """
    try:
        models = list_models()
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {'models': models}
