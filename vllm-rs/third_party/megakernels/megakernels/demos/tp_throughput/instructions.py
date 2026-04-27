from dataclasses import dataclass

from megakernels.instructions import Instruction


@dataclass
class ComputeInstruction(Instruction):
    @classmethod
    def tags(cls):
        return {"pool": "compute"}


@dataclass
class MemoryInstruction(Instruction):
    @classmethod
    def tags(cls):
        return {"pool": "memory"}


@dataclass
class NormInstruction(MemoryInstruction):
    layer_idx: int
    local_batch_indices: list[int]


@dataclass
class AttnNorm(NormInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProjResidual.opcode()


@dataclass
class MatMulInstruction(ComputeInstruction):
    layer_idx: int

    local_batch_block_idx: int
    local_output_block_idx: int
    global_batch_block_idx: int
    global_output_block_idx: int


@dataclass
class QKV_RopeAppend(MatMulInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> int:
        return AttnNorm.opcode()


@dataclass
class AttentionPrefill(ComputeInstruction):
    layer_idx: int
    prefill_seq_idx: int
    prefill_block_idx: int
    prefill_token_offset: int
    kv_head_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> int:
        return QKV_RopeAppend.opcode()


@dataclass
class AttentionDecode(ComputeInstruction):
    layer_idx: int
    # data: list[tuple[int, int]]
    # list of (seq_idx, kv_head_idx) tuples
    data: list[int]

    @classmethod
    def opcode(cls) -> int:
        return 4

    @classmethod
    def prev_opcode(cls) -> int:
        return AttentionPrefill.opcode()


@dataclass
class O_ProjResidual(MatMulInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> int:
        return AttentionDecode.opcode()


@dataclass
class MLP_Norm(NormInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> int:
        return O_ProjResidual.opcode()


@dataclass
class GateSilu(MatMulInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> int:
        return MLP_Norm.opcode()


@dataclass
class UpMatMul(MatMulInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 8

    @classmethod
    def prev_opcode(cls) -> int:
        return GateSilu.opcode()


@dataclass
class DownProjResidual(MatMulInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 9

    @classmethod
    def prev_opcode(cls) -> int:
        return UpMatMul.opcode()


@dataclass
class LM_Head_Norm(NormInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 10

    @classmethod
    def prev_opcode(cls) -> int:
        return DownProjResidual.opcode()


@dataclass
class LM_Head(MatMulInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 11

    @classmethod
    def prev_opcode(cls) -> int:
        return LM_Head_Norm.opcode()


@dataclass
class IncBarrier(MemoryInstruction):
    @classmethod
    def opcode(cls) -> int:
        return 12

    @classmethod
    def prev_opcode(cls) -> int:
        return 0  # No previous instruction needed


@dataclass
class Die(ComputeInstruction):
    @classmethod
    def opcode(cls) -> int:
        return -1

    @classmethod
    def prev_opcode(cls) -> int:
        return 0  # No previous instruction needed


@dataclass
class AllDeviceBarrier(MemoryInstruction):
    layer_idx: int
    bar_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 13

    @classmethod
    def prev_opcode(cls) -> int:
        return 0  # No previous instruction needed