"""Plain-language terms for running the contributor agent."""

TERMS = """\
/compute terms (LAN pool)

You are lending a fraction of this Mac's GPU time, chosen by you, to a
shared pipeline-parallel training pool. You can stop at any time. Stop
drains after the current step.

You earn usage credits in FLOPs, one-for-one with the work this Mac
actually does. Take spends your credits, up to the FLOP budget you set
for a job.

There is no money in this pool. Credits are not transferable off this
coordinator. Work is checked (canary, sampled replay). A coordinator
admin may flag or ban accounts that abuse the pool.

The dataset you submit stays on the coordinator machine. Do not submit
data you are not allowed to share with the people on this LAN.
"""
