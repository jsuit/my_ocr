from typing import Any
import torch


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, inp: Any) -> Any:
        pass


class ImplementsInterface(torch.nn.Module):
    def forward(self, inp: Any) -> Any:
        if isinstance(inp, torch.Tensor):
            return torch.max(inp, dim=0)

        return inp


class ModWithList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.ModuleList([ImplementsInterface()])

    def forward(self, x: torch.Tensor, idx: int) -> Any:
        value: ModuleInterface = self.l[idx]
        return value.forward(x)


m = torch.jit.script(ModWithList())
m.eval()
