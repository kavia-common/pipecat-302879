#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer-based, non-blocking conversation summarization.

This module provides a SummarizationObserver that listens to transcript updates
flowing through a Pipecat pipeline and maintains a continuously updated summary.

Design goals:
- Non-blocking: never stall the real-time pipeline. Summarization runs in an
  independent asyncio Task (queued via the pipeline's TaskObserver proxy).
- Minimal coupling: implemented as a regular BaseObserver; it can be attached to
  any PipelineTask via the `observers=[...]` argument.
- Safe boundaries: triggers on turn boundaries (best-effort) and/or transcript
  update frames.

Notes:
- The observer can be used with any custom async `summarizer` coroutine.
  If none is provided, the observer defaults to an extremely lightweight local
  summarizer that just compacts recent dialogue into a short rolling summary.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


@dataclass
class TranscriptLine:
    """A normalized transcript line."""

    role: str
    content: str


SummarizerFn = Callable[[str, List[TranscriptLine]], Awaitable[str]]


async def _default_summarizer(previous_summary: str, new_lines: List[TranscriptLine]) -> str:
    """Very small local summarizer.

    This exists so the observer works "out of the box" without adding service
    dependencies. It simply maintains a rolling summary string that:
    - keeps the previous summary
    - appends up to a few recent lines compactly
    - truncates to a sane maximum length

    Args:
        previous_summary: Current summary string.
        new_lines: Newly arrived transcript lines to incorporate.

    Returns:
        Updated summary string.
    """
    # Keep it short and deterministic.
    max_chars = 1200
    max_new = 12

    pieces: List[str] = []
    if previous_summary.strip():
        pieces.append(previous_summary.strip())

    # Add a compact "recent" section.
    compact = []
    for line in new_lines[-max_new:]:
        role = "User" if line.role == "user" else "Assistant"
        text = " ".join(line.content.split())
        compact.append(f"{role}: {text}")
    if compact:
        pieces.append("Recent:\n" + "\n".join(compact))

    out = "\n\n".join(pieces).strip()
    if len(out) > max_chars:
        out = out[-max_chars:]
        # Avoid slicing into the middle of a line too much.
        nl = out.find("\n")
        if 0 <= nl <= 60:
            out = out[nl + 1 :].lstrip()
    return out


class SummarizationObserver(BaseObserver):
    """Maintains a conversation summary based on transcript updates.

    The observer listens for TranscriptionUpdateFrame frames and collects
    normalized user/assistant text. At turn boundaries it schedules a background
    summarization job that updates `self.summary` and emits an optional callback.

    The summarization job is strictly best-effort:
    - at most one job runs at a time
    - multiple triggers coalesce into one summarization run
    """

    def __init__(
        self,
        *,
        summarizer: Optional[SummarizerFn] = None,
        on_summary_updated: Optional[Callable[[str], Awaitable[None]] | Callable[[str], None]] = None,
        summarize_on_turn_end: bool = True,
        summarize_on_transcript_update: bool = False,
        max_pending_lines: int = 200,
        **kwargs,
    ):
        """Initialize the SummarizationObserver.

        Args:
            summarizer: Async callable (previous_summary, new_lines) -> new_summary.
                If omitted, a tiny local summarizer is used.
            on_summary_updated: Optional callback invoked after summary is updated.
                Can be sync or async.
            summarize_on_turn_end: Trigger summarization when user/bot turn ends
                (UserStoppedSpeakingFrame or BotStoppedSpeakingFrame).
            summarize_on_transcript_update: Trigger summarization on each
                TranscriptionUpdateFrame (not recommended for realtime).
            max_pending_lines: Bound for buffered transcript lines waiting to be
                summarized (best-effort memory cap).
            **kwargs: Passed through to BaseObserver.
        """
        super().__init__(**kwargs)
        self._summarizer: SummarizerFn = summarizer or _default_summarizer
        self._on_summary_updated = on_summary_updated
        self._summarize_on_turn_end = summarize_on_turn_end
        self._summarize_on_transcript_update = summarize_on_transcript_update
        self._max_pending_lines = max_pending_lines

        self.summary: str = ""
        self._pending_lines: List[TranscriptLine] = []

        # Background task orchestration
        self._summarize_task: Optional[asyncio.Task] = None
        self._summarize_requested = False
        self._lock = asyncio.Lock()

    def get_summary(self) -> str:
        """Return the current summary (best-effort snapshot)."""
        return self.summary

    async def on_push_frame(self, data: FramePushed):
        """Handle incoming frames and schedule summarization.

        Args:
            data: The frame push event data.
        """
        frame = data.frame

        if isinstance(frame, TranscriptionUpdateFrame):
            self._ingest_transcription_update(frame)
            if self._summarize_on_transcript_update:
                self._request_summarization()

        if self._summarize_on_turn_end and isinstance(
            frame, (UserStoppedSpeakingFrame, BotStoppedSpeakingFrame)
        ):
            # Turn boundary: best time to summarize without spamming.
            self._request_summarization()

    def _ingest_transcription_update(self, frame: TranscriptionUpdateFrame) -> None:
        """Collect normalized transcript lines from update frames."""
        new_lines: List[TranscriptLine] = []
        for msg in frame.messages:
            # Only accept normalized user/assistant text messages.
            if isinstance(msg, TranscriptionMessage):
                role = msg.role
                content = msg.content or ""
                if role in ("user", "assistant") and content.strip():
                    new_lines.append(TranscriptLine(role=role, content=content.strip()))

        if not new_lines:
            return

        self._pending_lines.extend(new_lines)
        # Memory bound - keep the most recent items.
        if len(self._pending_lines) > self._max_pending_lines:
            self._pending_lines = self._pending_lines[-self._max_pending_lines :]

    def _request_summarization(self) -> None:
        """Request a background summarization run (coalescing multiple triggers)."""
        self._summarize_requested = True
        if self._summarize_task is None or self._summarize_task.done():
            self._summarize_task = asyncio.create_task(self._summarize_loop())

    async def _summarize_loop(self) -> None:
        """Run summarization until no more requests are pending."""
        # Coalesce bursts: small delay lets multiple frames accumulate.
        await asyncio.sleep(0)

        while self._summarize_requested:
            self._summarize_requested = False
            try:
                await self._summarize_once()
            except Exception as e:
                # Never crash the pipeline observer task.
                logger.exception(f"SummarizationObserver summarization error: {e}")

            # Yield to allow more frame delivery / coalescing.
            await asyncio.sleep(0)

    async def _summarize_once(self) -> None:
        """Perform one summarization update from buffered transcript lines."""
        async with self._lock:
            if not self._pending_lines:
                return
            new_lines = list(self._pending_lines)
            self._pending_lines.clear()

        new_summary = await self._summarizer(self.summary, new_lines)
        self.summary = new_summary

        # Notify callback (best-effort, sync or async).
        if self._on_summary_updated:
            try:
                result = self._on_summary_updated(self.summary)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.exception(f"SummarizationObserver on_summary_updated error: {e}")
