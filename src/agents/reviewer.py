from src.agents import Agent


class Reviewer(Agent):
    ROLE = "reviewer"
    STAGE = "reviewer"

    def prompt(self, task_description: str, code: str, test_output: str) -> str:
        return self.build_prompt(task_description=task_description, code=code, test_output=test_output)

    @staticmethod
    def parse(text: str) -> str:
        return (text or "").strip()
