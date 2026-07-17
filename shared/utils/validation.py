import re
from fastapi import HTTPException, status

def validate_name(name: str, field_name: str = "Name") -> str:
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} cannot be empty."
        )
    if name.startswith(" ") or name.endswith(" "):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} cannot start or end with spaces."
        )
    cleaned = name
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} cannot consist only of whitespace."
        )
    if len(cleaned) < 1 or len(cleaned) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must be between 1 and 50 characters."
        )
    # Match standard name characters: Unicode letters, spaces, hyphens, and apostrophes
    if not re.match(r"^[a-zA-ZÀ-ÿ\s'-]+$", cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} can only contain letters, spaces, hyphens, or apostrophes."
        )
    return cleaned

def validate_employee_code(code: str) -> str:
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee code cannot be empty."
        )
    if code.startswith(" ") or code.endswith(" "):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee code cannot start or end with spaces."
        )
    cleaned = code
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee code cannot consist only of whitespace."
        )
    if len(cleaned) < 2 or len(cleaned) > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee code must be between 2 and 30 characters."
        )
    # Match alphanumeric characters, hyphens, or underscores
    if not re.match(r"^[a-zA-Z0-9_-]+$", cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee code can only contain alphanumeric characters, hyphens, or underscores."
        )
    return cleaned
