import pytest

from slashcompute.metering import (
    UsageRecorder, forward_flops, max_stage_step_flops, stage_step_flops,
)
from slashcompute.pipeline.model_profile import ModelProfile
from slashcompute.pipeline.schedule import StepStats


def _profile(n=4):
    return ModelProfile(model="m", num_layers=n, hidden_size=8, vocab_size=10,
                        tie_word_embeddings=True, layer_bytes=(100,) * n, embed_bytes=40,
                        head_bytes=50, layer_params=1000, head_params=80)


def test_forward_flops_formula():
    p = _profile()
    # 2 layers * (2*1000 + 2*seq(5)*hidden(8)) per token, 3 tokens
    assert forward_flops(p, 2, tokens=3, seq_len=5, include_head=False) == 2 * (2000 + 80) * 3
    assert forward_flops(p, 1, 1, 5, include_head=True) == (2000 + 80) + 160


def test_stage_flops_sum_is_partition_independent_except_recompute():
    p = _profile()
    whole = stage_step_flops(p, 0, 4, 10, 10, is_last=True)
    assert whole == pytest.approx(2 * forward_flops(p, 4, 10, 10, True))
    split = stage_step_flops(p, 0, 2, 10, 10, False) + stage_step_flops(p, 2, 4, 10, 10, True)
    # non-last stages pay one extra forward for recomputation
    assert split == pytest.approx(whole + forward_flops(p, 2, 10, 10, False))


def test_recorder_accumulates():
    p = _profile()
    r = UsageRecorder(p, 2, 4)
    s = StepStats(step=1, loss=1.0, tokens_processed=10, loss_tokens=8, seq_len=10, wall_s=2.0,
                  busy_s=1.5, peak_mem_bytes=500, resident_mem_bytes=300, in_digest="a",
                  out_digest="b")
    u = r.sample(s)
    assert u.mem_byte_seconds == 600.0 and u.busy_s == 1.5
    assert u.flops == stage_step_flops(p, 2, 4, 10, 10, True)
    r.sample(s)
    assert r.steps == 2 and r.total_flops == 2 * u.flops


@pytest.mark.parametrize("layers", [(0, 4), (0, 2), (2, 4)])
def test_no_honest_step_reports_more_than_the_stage_ceiling(layers):
    """The coordinator bills a step up to this ceiling, so what a stage meters for any batch
    the job allows (rows padded to at most max_seq_len) must fit under it, with no slack."""
    p, batch_size, max_seq_len = _profile(), 4, 32
    ceiling = max_stage_step_flops(p, *layers, batch_size, max_seq_len)
    reported = [
        UsageRecorder(p, *layers).sample(StepStats(
            step=1, loss=None, tokens_processed=batch_size * seq, loss_tokens=0, seq_len=seq,
            wall_s=1.0, busy_s=1.0, peak_mem_bytes=1, resident_mem_bytes=1, in_digest="a",
            out_digest="b")).flops
        for seq in range(1, max_seq_len + 1)]
    assert max(reported) == reported[-1] == ceiling
    assert reported == sorted(reported)
