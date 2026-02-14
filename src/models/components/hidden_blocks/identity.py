from src.models.components.base import ForwardState, HiddenBlock


class Identity(HiddenBlock):
    def forward(self, x: ForwardState) -> ForwardState:
        return x
