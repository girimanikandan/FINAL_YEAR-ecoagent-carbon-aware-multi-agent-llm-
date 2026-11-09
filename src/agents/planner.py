from src.agents import Agent


class Planner(Agent):
    ROLE = "planner"
    STAGE = "planner"

    def prompt(self, task_description: str) -> str:
        return self.build_prompt(task_description=task_description)

    @staticmethod
    def parse(text: str) -> str:
        return (text or "").strip()
