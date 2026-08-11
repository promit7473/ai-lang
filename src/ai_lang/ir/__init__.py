from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class IROp(Enum):
    PERCEIVE = "perceive"
    REASON = "reason"
    PLAN = "plan"
    CONTROL = "control"
    VERIFY = "verify"
    OUTPUT = "output"
    COMPUTE = "compute"
    EVALUATE = "evaluate"


class ObjectiveType(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


class ConstraintOp(Enum):
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    EQ = "="
    NE = "!="


@dataclass
class InputDecl:
    name: str
    type_name: str
    description: str = ""


@dataclass
class Assignment:
    target: str
    func: str
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


@dataclass
class Objective:
    type: ObjectiveType
    metric: str
    weight: float = 1.0


@dataclass
class Constraint:
    metric: str
    op: ConstraintOp
    value: Any
    unit: str = ""


@dataclass
class VerifyCheck:
    name: str
    condition: Optional[str] = None


@dataclass
class OutputDecl:
    name: str
    type_name: str


@dataclass
class Block:
    op: IROp
    statements: list = field(default_factory=list)
    objectives: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    outputs: list = field(default_factory=list)


@dataclass
class TaskIR:
    name: str
    inputs: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    outputs: list = field(default_factory=list)

    def get_block(self, op: IROp) -> Optional[Block]:
        for b in self.blocks:
            if b.op == op:
                return b
        return None
