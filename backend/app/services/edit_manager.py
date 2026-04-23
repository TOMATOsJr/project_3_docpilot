from __future__ import annotations

from dataclasses import dataclass, field
from difflib import ndiff
from uuid import UUID, uuid4

from app.core.models import DiffLine, EditProposal


@dataclass(slots=True)
class EditCommand:
    document_id: UUID
    instruction: str
    original_text: str
    proposed_text: str
    command_id: UUID = field(default_factory=uuid4)

    def to_proposal(self) -> EditProposal:
        diff = [DiffLine(kind=_diff_kind(line), content=line[2:]) for line in ndiff(self.original_text.splitlines(), self.proposed_text.splitlines()) if not line.startswith("? ")]
        return EditProposal(
            command_id=self.command_id,
            document_id=self.document_id,
            instruction=self.instruction,
            diff=diff,
            original_text=self.original_text,
            proposed_text=self.proposed_text,
        )


def _diff_kind(line: str) -> str:
    if line.startswith("+ "):
        return "insert"
    if line.startswith("- "):
        return "delete"
    return "equal"
