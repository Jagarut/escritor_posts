from typing import List, Dict

class StoryManager:
    """
    Manages story versions and refinements.
    """
    def __init__(self):
        self.versions: List[Dict[str, str]] = []  # [{text: "...", feedback: "..."}]

    def add_version(self, text: str, feedback: str = "Initial draft") -> None:
        """
        Adds a new version to the history.
        
        Args:
            text (str): Story text.
            feedback (str): User feedback for this version.
        """
        self.versions.append({"text": text, "feedback": feedback})

    def get_latest_version(self) -> str:
        """Returns the latest version of the story."""
        return self.versions[-1]["text"] if self.versions else ""

    def get_version_history(self) -> List[Dict[str, str]]:
        """Returns full version history."""
        return self.versions