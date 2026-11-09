from src.agents import Agent, extract_code

NO_PLAN = "No plan provided. Implement directly."


class Coder(Agent):
    ROLE = "coder"
    STAGE = "coder"

    def prompt(self, task_description: str, plan: str = "") -> str:
        return self.build_prompt(task_description=task_description, plan=plan or NO_PLAN)

    @staticmethod
    def parse(text: str) -> str:
        return extract_code(text)
