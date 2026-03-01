"""Language information API routes."""

from fastapi import APIRouter

from vaanidub.api.schemas import LanguageListResponse, LanguageResponse
from vaanidub.constants import LANGUAGES

router = APIRouter(tags=["languages"])


@router.get("/languages", response_model=LanguageListResponse)
async def list_languages():
    """List all supported languages."""
    return LanguageListResponse(
        languages=[
            LanguageResponse(
                code=info.code,
                name=info.name,
                native_name=info.native_name,
                script=info.script,
                tts_providers=list(info.tts_providers),
            )
            for info in LANGUAGES.values()
        ]
    )
