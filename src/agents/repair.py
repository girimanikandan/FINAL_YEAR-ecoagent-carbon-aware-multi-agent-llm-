from src.agents import Agent, extract_code


class Repair(Agent):
    ROLE = "repair"
    STAGE = "coder"      # Coder-Repair is a second pass of the coder stage

    def prompt(self, task_description: str, code: str, review: str, test_output: str) -> str:
        return self.build_prompt(task_description=task_description, code=code, review=review, test_output=test_output)

    @staticmethod
    def parse(text: str) -> str:
        return extract_code(text)
