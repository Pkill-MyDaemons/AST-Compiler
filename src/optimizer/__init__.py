"""AST optimization passes."""
from .passes import ConstantFoldingPass, IdentityEliminationPass, DeadCodeEliminationPass
from ..unified_ast.nodes import Module


def optimize(module: Module, passes=None) -> Module:
    """Run all optimization passes (or the supplied list) and return the result."""
    if passes is None:
        passes = [ConstantFoldingPass(), IdentityEliminationPass(), DeadCodeEliminationPass()]
    for p in passes:
        module = p.run(module)
    return module


__all__ = [
    "optimize",
    "ConstantFoldingPass",
    "IdentityEliminationPass",
    "DeadCodeEliminationPass",
]
