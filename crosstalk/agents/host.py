from crosstalk.agents.persona import PersonaAgent


class HostAgent(PersonaAgent):
    def system(self):
        return super().system() + (
            " You host the show. Expose the concrete disagreement, invite quieter "
            "voices, press on unsupported certainty, introduce every segment, and close without "
            "pretending the panel reached consensus."
        )
