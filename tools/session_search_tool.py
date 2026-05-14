#!/usr/bin/env python3
"""
Session Search Tool - Long-Term Conversation Recall

Searches past session transcripts in SQLite via FTS5. Keyword search defaults
to fast snippet/context hits without any LLM call; callers can opt into focused
LLM summaries with mode="summary" when deeper recall is worth the latency.

Flow:
  1. FTS5 search finds matching messages ranked by relevance
  2. Groups by session, takes the top N unique sessions (default 3)
  3. Fast mode returns snippets and nearby context immediately
  4. Summary mode loads each session, truncates around matches, and calls an LLM
  5. Returns per-session hits/summaries with metadata
"""

import asyncio
import concurrent.futures
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union

from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
MAX_SESSION_CHARS = 100_000


# Default mode is summary unless the user opts into a different one via
# ``auxiliary.session_search.default_mode`` in ~/.hermes/config.yaml. Lives
# alongside the other session_search-scoped knobs (provider, model,
# max_concurrency) — "auxiliary" started as aux-LLM routing but in practice
# groups per-tool config by tool name. Only ``fast`` and ``summary`` are
# valid defaults — guided requires anchors and can't be used standalone.
# Wrapped in lru_cache so the YAML read happens at most once per process;
# the CLI / TUI is the typical caller and config changes need a restart
# anyway.
_VALID_DEFAULT_MODES = ("fast", "summary")


def _resolve_user_default_mode() -> str:
    """Look up ``auxiliary.session_search.default_mode`` from ~/.hermes/config.yaml.

    Returns "summary" if unset, invalid, or the config loader is unavailable
    (e.g. tests, tools loaded outside the CLI). Logs a one-time warning on
    invalid values so users get feedback when they typo their config.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config() or {}
    except ImportError:
        logging.debug("hermes_cli.config not available; default_mode falls back to 'summary'")
        return "summary"
    except Exception as e:
        logging.debug("Failed to load config for session_search default_mode: %s", e, exc_info=True)
        return "summary"

    raw = (
        config.get("auxiliary", {})
        .get("session_search", {})
        .get("default_mode")
    )
    if raw is None:
        return "summary"
    if not isinstance(raw, str):
        logging.warning(
            "auxiliary.session_search.default_mode in config.yaml must be a string, got %r — falling back to 'summary'",
            raw,
        )
        return "summary"
    normalised = raw.strip().lower()
    if normalised not in _VALID_DEFAULT_MODES:
        logging.warning(
            "auxiliary.session_search.default_mode=%r is not one of %s — falling back to 'summary'. "
            "(guided requires anchors and cannot be a default.)",
            raw, _VALID_DEFAULT_MODES,
        )
        return "summary"
    return normalised


# Process-level cache so repeated session_search calls don't re-read YAML.
# Cleared by tests via _resolve_user_default_mode.cache_clear() when needed.
import functools  # noqa: E402  — local to the cache wrap
_resolve_user_default_mode = functools.lru_cache(maxsize=1)(_resolve_user_default_mode)
MAX_SUMMARY_TOKENS = 10000


def _get_session_search_max_concurrency(default: int = 3) -> int:
    """Read auxiliary.session_search.max_concurrency with sane bounds."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except ImportError:
        return default
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get("session_search", {}) if isinstance(aux, dict) else {}
    if not isinstance(task_config, dict):
        return default
    raw = task_config.get("max_concurrency")
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, 5))


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Convert a Unix timestamp (float/int) or ISO string to a human-readable date.

    Returns "unknown" for None, str(ts) if conversion fails.
    """
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        if isinstance(ts, str):
            if ts.replace(".", "").replace("-", "").isdigit():
                from datetime import datetime
                dt = datetime.fromtimestamp(float(ts))
                return dt.strftime("%B %d, %Y at %I:%M %p")
            return ts
    except (ValueError, OSError, OverflowError) as e:
        # Log specific errors for debugging while gracefully handling edge cases
        logging.debug("Failed to format timestamp %s: %s", ts, e, exc_info=True)
    except Exception as e:
        logging.debug("Unexpected error formatting timestamp %s: %s", ts, e, exc_info=True)
    return str(ts)


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """Format session messages into a readable transcript for summarization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_name = msg.get("tool_name")

        if role == "TOOL" and tool_name:
            # Truncate long tool outputs
            if len(content) > 500:
                content = content[:250] + "\n...[truncated]...\n" + content[-250:]
            parts.append(f"[TOOL:{tool_name}]: {content}")
        elif role == "ASSISTANT":
            # Include tool call names if present
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                tc_names = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name") or tc.get("function", {}).get("name", "?")
                        tc_names.append(name)
                if tc_names:
                    parts.append(f"[ASSISTANT]: [Called: {', '.join(tc_names)}]")
                if content:
                    parts.append(f"[ASSISTANT]: {content}")
            else:
                parts.append(f"[ASSISTANT]: {content}")
        else:
            parts.append(f"[{role}]: {content}")

    return "\n\n".join(parts)


def _truncate_around_matches(
    full_text: str, query: str, max_chars: int = MAX_SESSION_CHARS
) -> str:
    """
    Truncate a conversation transcript to *max_chars*, choosing a window
    that maximises coverage of positions where the *query* actually appears.

    Strategy (in priority order):
    1. Try to find the full query as a phrase (case-insensitive).
    2. If no phrase hit, look for positions where all query terms appear
       within a 200-char proximity window (co-occurrence).
    3. Fall back to individual term positions.

    Once candidate positions are collected the function picks the window
    start that covers the most of them.
    """
    if len(full_text) <= max_chars:
        return full_text

    text_lower = full_text.lower()
    query_lower = query.lower().strip()
    match_positions: list[int] = []

    # --- 1. Full-phrase search ------------------------------------------------
    phrase_pat = re.compile(re.escape(query_lower))
    match_positions = [m.start() for m in phrase_pat.finditer(text_lower)]

    # --- 2. Proximity co-occurrence of all terms (within 200 chars) -----------
    if not match_positions:
        terms = query_lower.split()
        if len(terms) > 1:
            # Collect every occurrence of each term
            term_positions: dict[str, list[int]] = {}
            for t in terms:
                term_positions[t] = [
                    m.start() for m in re.finditer(re.escape(t), text_lower)
                ]
            # Slide through positions of the rarest term and check proximity
            rarest = min(terms, key=lambda t: len(term_positions.get(t, [])))
            for pos in term_positions.get(rarest, []):
                if all(
                    any(abs(p - pos) < 200 for p in term_positions.get(t, []))
                    for t in terms
                    if t != rarest
                ):
                    match_positions.append(pos)

    # --- 3. Individual term positions (last resort) ---------------------------
    if not match_positions:
        terms = query_lower.split()
        for t in terms:
            for m in re.finditer(re.escape(t), text_lower):
                match_positions.append(m.start())

    if not match_positions:
        # Nothing at all — take from the start
        truncated = full_text[:max_chars]
        suffix = "\n\n...[later conversation truncated]..." if max_chars < len(full_text) else ""
        return truncated + suffix

    # --- Pick window that covers the most match positions ---------------------
    match_positions.sort()

    best_start = 0
    best_count = 0
    for candidate in match_positions:
        ws = max(0, candidate - max_chars // 4)  # bias: 25% before, 75% after
        we = ws + max_chars
        if we > len(full_text):
            ws = max(0, len(full_text) - max_chars)
            we = len(full_text)
        count = sum(1 for p in match_positions if ws <= p < we)
        if count > best_count:
            best_count = count
            best_start = ws

    start = best_start
    end = min(len(full_text), start + max_chars)

    truncated = full_text[start:end]
    prefix = "...[earlier conversation truncated]...\n\n" if start > 0 else ""
    suffix = "\n\n...[later conversation truncated]..." if end < len(full_text) else ""
    return prefix + truncated + suffix


async def _summarize_session(
    conversation_text: str, query: str, session_meta: Dict[str, Any]
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Summarize a single session conversation focused on the search query.

    Returns ``(content, usage)`` where ``usage`` is a dict with
    ``{model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens}``
    parsed from the aux LLM response, or ``None`` when the model didn't surface
    usage data. The usage dict lets callers attribute the cost of summary-mode
    aux calls back to the parent session — without this, summary-mode spend is
    invisible to per-session accounting.
    """
    system_prompt = (
        "You are reviewing a past conversation transcript to help recall what happened. "
        "Summarize the conversation with a focus on the search topic. Include:\n"
        "1. What the user asked about or wanted to accomplish\n"
        "2. What actions were taken and what the outcomes were\n"
        "3. Key decisions, solutions found, or conclusions reached\n"
        "4. Any specific commands, files, URLs, or technical details that were important\n"
        "5. Anything left unresolved or notable\n\n"
        "Be thorough but concise. Preserve specific details (commands, paths, error messages) "
        "that would be useful to recall. Write in past tense as a factual recap."
    )

    source = session_meta.get("source", "unknown")
    started = _format_timestamp(session_meta.get("started_at"))

    user_prompt = (
        f"Search topic: {query}\n"
        f"Session source: {source}\n"
        f"Session date: {started}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{conversation_text}\n\n"
        f"Summarize this conversation with focus on: {query}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await async_call_llm(
                task="session_search",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_SUMMARY_TOKENS,
            )
            content = extract_content_or_reasoning(response)
            usage = _extract_aux_usage(response)
            if content:
                return content, usage
            # Reasoning-only / empty — let the retry loop handle it
            logging.warning("Session search LLM returned empty content (attempt %d/%d)", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
                continue
            return content, usage
        except RuntimeError:
            logging.warning("No auxiliary model available for session summarization")
            return None, None
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                logging.warning(
                    "Session summarization failed after %d attempts: %s",
                    max_retries,
                    e,
                    exc_info=True,
                )
                return None, None


def _extract_aux_usage(response: Any) -> Optional[Dict[str, Any]]:
    """Pull usage data off an aux LLM response, normalising provider variants.

    Returns ``None`` when the response carries no usage info (test mocks,
    providers that don't surface it). Returns a dict with the fields we care
    about for cost attribution otherwise. Reads both OpenAI-style
    (``prompt_tokens``/``completion_tokens``) and Anthropic-style
    (``input_tokens``/``output_tokens``) usage shapes.
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    # Provider variants — read whichever is populated.
    input_tokens = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None)
        or 0
    )
    output_tokens = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or 0
    )
    # Anthropic prompt-caching fields.
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", None) or 0
    # OpenAI-style cached tokens may live under prompt_tokens_details.
    if not cache_read:
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cache_read = getattr(details, "cached_tokens", 0) or 0
    model = getattr(response, "model", None)
    return {
        "model": model,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cache_read_tokens": int(cache_read or 0),
        "cache_creation_tokens": int(cache_create or 0),
    }


# Sources that are excluded from session browsing/searching by default.
# Third-party integrations (Paperclip agents, etc.) tag their sessions with
# HERMES_SESSION_SOURCE=tool so they don't clutter the user's session history.
_HIDDEN_SESSION_SOURCES = ("tool",)


def _list_recent_sessions(db, limit: int, current_session_id: str = None) -> str:
    """Return metadata for the most recent sessions (no LLM calls)."""
    try:
        sessions = db.list_sessions_rich(
            limit=limit + 5,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            order_by_last_active=True,
        )  # fetch extra to skip current

        # Resolve current session lineage to exclude it
        current_root = None
        if current_session_id:
            try:
                sid = current_session_id
                visited = set()
                current_root = current_session_id
                while sid and sid not in visited:
                    visited.add(sid)
                    current_root = sid
                    s = db.get_session(sid)
                    parent = s.get("parent_session_id") if s else None
                    sid = parent if parent else None
            except Exception:
                current_root = current_session_id

        results = []
        for s in sessions:
            sid = s.get("id", "")
            if current_root and (sid == current_root or sid == current_session_id):
                continue
            # Skip child/delegation sessions (they have parent_session_id)
            if s.get("parent_session_id"):
                continue
            results.append({
                "session_id": sid,
                "title": s.get("title") or None,
                "source": s.get("source", ""),
                "started_at": s.get("started_at", ""),
                "last_active": s.get("last_active", ""),
                "message_count": s.get("message_count", 0),
                "preview": s.get("preview", ""),
            })
            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "mode": "recent",
            "results": results,
            "count": len(results),
            "message": f"Showing {len(results)} most recent sessions. Use a keyword query to search specific topics.",
        }, ensure_ascii=False)
    except Exception as e:
        logging.error("Error listing recent sessions: %s", e, exc_info=True)
        return tool_error(f"Failed to list recent sessions: {e}", success=False)


def _guided_drill_down(
    db,
    session_id: str,
    around_message_id,
    window: int,
    current_session_id: str = None,
    anchors: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Anchored drill-down for ``mode='guided'`` of ``session_search``.

    Returns a JSON string carrying one or more windows of messages — each
    centred on a specific message id in a specific session. No FTS5, no
    auxiliary LLM, no 100k-char truncation — N indexed DB lookups (where
    N = number of anchors).

    Two input shapes (use one):

      * **Single anchor** (back-compat): pass ``session_id`` and
        ``around_message_id`` directly. Internally normalised to a single-
        element ``anchors`` list. Response always carries ``windows``
        as a list, plus the legacy single-anchor fields at the top level
        when there's exactly one anchor.

      * **Multi-anchor**: pass ``anchors=[{"session_id":..., "around_message_id":...}, ...]``.
        The agent picks the most promising K hits from a wider fast call
        and drills into all of them at once — same conversation in the
        steering loop, more context per turn.

    Each anchor is validated independently. Per-anchor failures (missing
    session, anchor not in session, current-lineage rejection) become
    error entries inside the response's ``windows`` list rather than
    aborting the whole call. ``window`` is shared across all anchors
    and clamped to ``[1, 20]`` (silent, matches the existing limit-clamp
    pattern).
    """
    # 1. Normalise inputs into a single ``anchors`` list. Three shapes:
    #    (a) anchors= parameter is set (preferred for multi-anchor)
    #    (b) session_id + around_message_id (single-anchor back-compat)
    #    (c) neither set → user-facing error
    if anchors:
        if not isinstance(anchors, list):
            return tool_error(
                "guided mode: 'anchors' must be a list of {session_id, around_message_id} dicts",
                success=False,
            )
        normalised_anchors = anchors
    elif session_id or around_message_id is not None:
        normalised_anchors = [{
            "session_id": session_id,
            "around_message_id": around_message_id,
        }]
    else:
        return tool_error(
            "guided mode requires either anchors=[...] or session_id+around_message_id "
            "(use match_message_id+session_id from a prior fast-mode hit)",
            success=False,
        )

    if len(normalised_anchors) == 0:
        return tool_error(
            "guided mode: anchors list is empty (pass at least one {session_id, around_message_id})",
            success=False,
        )

    # 2. Window clamp (shared across all anchors). Matches the existing
    #    limit-clamp pattern (silent).
    if not isinstance(window, int):
        try:
            window = int(window)
        except (TypeError, ValueError):
            window = 5
    window = max(1, min(window, 20))

    # 3. Helper: resolve to lineage root (used by the current-lineage
    #    rejection check below).
    def _resolve_to_parent(sid: str) -> str:
        visited = set()
        cur = sid
        while cur and cur not in visited:
            visited.add(cur)
            try:
                meta = db.get_session(cur)
                if not meta:
                    break
                parent = meta.get("parent_session_id")
                if parent:
                    cur = parent
                else:
                    break
            except Exception as e:
                logging.debug("Error resolving parent for %s: %s", cur, e, exc_info=True)
                break
        return cur

    current_root = _resolve_to_parent(current_session_id) if current_session_id else None

    # 4. Drill into each anchor. Per-anchor errors are recorded inline
    #    rather than aborting the whole call — the agent can still use
    #    successful drills even if one anchor was malformed.
    windows_out: List[Dict[str, Any]] = []
    for raw_anchor in normalised_anchors:
        if not isinstance(raw_anchor, dict):
            windows_out.append({
                "success": False,
                "error": "anchor must be a dict with session_id + around_message_id",
            })
            continue

        a_sid = raw_anchor.get("session_id")
        a_msg = raw_anchor.get("around_message_id")

        if not a_sid or not isinstance(a_sid, str) or not a_sid.strip():
            windows_out.append({
                "success": False,
                "error": "anchor missing session_id",
                "anchor": raw_anchor,
            })
            continue
        a_sid = a_sid.strip()

        try:
            a_msg_id = int(a_msg)
        except (TypeError, ValueError):
            windows_out.append({
                "success": False,
                "error": "anchor missing or non-integer around_message_id",
                "anchor": raw_anchor,
            })
            continue

        # Current-lineage rejection: per-anchor, so other valid anchors
        # in a multi-anchor call still drill.
        if current_root:
            target_root = _resolve_to_parent(a_sid)
            if target_root and target_root == current_root:
                windows_out.append({
                    "success": False,
                    "error": "anchor rejects drill-down into the current session lineage — those messages are already in your active context",
                    "session_id": a_sid,
                    "around_message_id": a_msg_id,
                })
                continue

        # Session existence check.
        try:
            session_meta = db.get_session(a_sid) or {}
        except Exception as e:
            logging.debug("get_session failed for %s: %s", a_sid, e, exc_info=True)
            session_meta = {}
        if not session_meta:
            windows_out.append({
                "success": False,
                "error": f"session_id not found: {a_sid}",
                "session_id": a_sid,
                "around_message_id": a_msg_id,
            })
            continue

        # Fetch the window + bookends. ``get_anchored_view`` filters tool-response
        # noise from the window (anchor itself is preserved regardless of role)
        # and returns up to ``bookend`` user/assistant messages from the session
        # head and tail — but only when those slices don't overlap the window.
        # See SessionDB.get_anchored_view for the contract.
        try:
            view = db.get_anchored_view(a_sid, a_msg_id, window=window, bookend=3)
            messages = view.get("window") or []
            bookend_start = view.get("bookend_start") or []
            bookend_end = view.get("bookend_end") or []
        except Exception as e:
            logging.debug("get_anchored_view failed: %s", e, exc_info=True)
            windows_out.append({
                "success": False,
                "error": f"failed to load messages around {a_msg_id} in {a_sid}: {e}",
                "session_id": a_sid,
                "around_message_id": a_msg_id,
            })
            continue

        # Safety net: the agent (or memory, or a legacy caller) may pair a
        # parent/lineage-root session_id with a message_id that actually
        # lives in a descendant (child) session. Before this commit, fast
        # mode returned exactly that broken pair. We now emit the matching
        # raw sid in fast mode, but guided should remain forgiving for
        # callers that haven't updated yet.
        #
        # Recovery rule: locate the real owning session by message id; if
        # that session is in the same lineage as ``a_sid``, transparently
        # rebind and refetch. Record a warning so the rebind is visible.
        rebind_warning = None
        if not messages:
            owning = None
            # Prefer a helper if SessionDB exposes one (forward-compat).
            try:
                if hasattr(db, "get_session_id_for_message"):
                    owning = db.get_session_id_for_message(a_msg_id)
            except Exception as e:
                logging.debug("get_session_id_for_message failed: %s", e, exc_info=True)
                owning = None
            # Fallback: query through SessionDB._conn (the canonical connection).
            if not owning:
                try:
                    conn = getattr(db, "_conn", None)
                    if conn is not None:
                        row = conn.execute(
                            "SELECT session_id FROM messages WHERE id = ?",
                            (a_msg_id,),
                        ).fetchone()
                        # sqlite3.Row supports indexing; tuple fallback works too.
                        owning = row[0] if row else None
                except Exception as e:
                    logging.debug("owning-session lookup failed: %s", e, exc_info=True)
                    owning = None

            if owning and owning != a_sid:
                # Check same lineage (walk both up to roots).
                a_root = _resolve_to_parent(a_sid)
                o_root = _resolve_to_parent(owning)
                if a_root and o_root and a_root == o_root:
                    try:
                        rebind_view = db.get_anchored_view(
                            owning, a_msg_id, window=window, bookend=3
                        )
                        messages = rebind_view.get("window") or []
                        bookend_start = rebind_view.get("bookend_start") or []
                        bookend_end = rebind_view.get("bookend_end") or []
                    except Exception as e:
                        logging.debug("rebind get_anchored_view failed: %s", e, exc_info=True)
                        messages = []
                    if messages:
                        rebind_warning = (
                            f"around_message_id {a_msg_id} lives in {owning} "
                            f"(child of {a_sid}); rebound transparently"
                        )
                        # Re-fetch session_meta for the actual owning session.
                        try:
                            session_meta = db.get_session(owning) or session_meta
                        except Exception:
                            pass
                        a_sid = owning

        if not messages:
            windows_out.append({
                "success": False,
                "error": f"around_message_id {a_msg_id} not in session_id {a_sid}",
                "session_id": a_sid,
                "around_message_id": a_msg_id,
            })
            continue

        # Wrap with anchor flag + boundary counts.
        out_messages = []
        messages_before = 0
        messages_after = 0
        for m in messages:
            is_anchor = m.get("id") == a_msg_id
            if not is_anchor and m.get("id", 0) < a_msg_id:
                messages_before += 1
            elif not is_anchor:
                messages_after += 1
            entry = {
                "id": m.get("id"),
                "role": m.get("role"),
                "content": m.get("content"),
                "tool_name": m.get("tool_name"),
                "tool_calls": m.get("tool_calls") or None,
                "tool_call_id": m.get("tool_call_id"),
                "timestamp": m.get("timestamp"),
            }
            if is_anchor:
                entry["anchor"] = True
            # Strip None-valued optional fields to keep payload tight (keep
            # 'content' even if None, since absent-content is meaningful).
            entry = {k: v for k, v in entry.items() if v is not None or k in ("content",)}
            out_messages.append(entry)

        def _shape_bookend(m: Dict[str, Any]) -> Dict[str, Any]:
            entry = {
                "id": m.get("id"),
                "role": m.get("role"),
                "content": m.get("content"),
                "timestamp": m.get("timestamp"),
            }
            return {k: v for k, v in entry.items() if v is not None or k in ("content",)}

        out_bookend_start = [_shape_bookend(m) for m in bookend_start]
        out_bookend_end = [_shape_bookend(m) for m in bookend_end]

        success_entry = {
            "success": True,
            "session_id": a_sid,
            "around_message_id": a_msg_id,
            "session_meta": {
                "when": _format_timestamp(session_meta.get("started_at")),
                "source": session_meta.get("source"),
                "model": session_meta.get("model"),
                "title": session_meta.get("title"),
            },
            "messages": out_messages,
            "messages_before": messages_before,
            "messages_after": messages_after,
            "bookend_start": out_bookend_start,
            "bookend_end": out_bookend_end,
        }
        if rebind_warning:
            success_entry["warning"] = rebind_warning
        windows_out.append(success_entry)

    # 5. Top-level response shape. ``windows`` is always a list. For
    #    single-anchor calls (the common case), we mirror the legacy fields
    #    at the top level so existing callers / tests continue to work
    #    without branching on len(windows).
    response: Dict[str, Any] = {
        "success": True,
        "mode": "guided",
        "window": window,
        "windows": windows_out,
        "anchor_count": len(windows_out),
    }
    if len(windows_out) == 1:
        only = windows_out[0]
        if only.get("success"):
            response.update({
                "session_id": only["session_id"],
                "around_message_id": only["around_message_id"],
                "session_meta": only["session_meta"],
                "messages": only["messages"],
                "messages_before": only["messages_before"],
                "messages_after": only["messages_after"],
                "bookend_start": only.get("bookend_start", []),
                "bookend_end": only.get("bookend_end", []),
            })
            if only.get("warning"):
                response["warning"] = only["warning"]
        else:
            # Single-anchor failure: surface as a top-level tool_error so
            # callers don't have to dig into the windows array for the
            # error string. Keeps the legacy single-anchor failure shape.
            return tool_error(only.get("error", "guided drill-down failed"), success=False)

    return json.dumps(response, ensure_ascii=False)


def session_search(
    query: str = "",
    role_filter: str = None,
    limit: int = 3,
    db=None,
    current_session_id: str = None,
    mode: str = None,
    # Guided-mode-only parameters: anchored drill-down into one or more
    # session+message pairs. Required when mode='guided', ignored otherwise.
    # Use either the single-anchor pair (session_id + around_message_id) or
    # the multi-anchor list (anchors=[{session_id, around_message_id}, ...]).
    session_id: str = None,
    around_message_id: int = None,
    window: int = 5,
    anchors: list = None,
) -> str:
    """
    Search past sessions, or drill into a specific one.

    Modes:
      * fast    — FTS5 snippets + ±1 message context. Cheap discovery.
      * summary — fetch full session(s), truncate to 100k chars, run aux LLM
                  recap. Cross-session synthesis at ~30s tool-side cost.
      * guided  — anchored drill-down. Caller supplies session_id +
                  around_message_id (typically from a prior fast hit's
                  match_message_id field) and gets a window of messages
                  around the anchor with no LLM call and no truncation.
    """
    if db is None:
        try:
            from hermes_state import SessionDB

            db = SessionDB()
        except Exception:
            logging.debug("SessionDB unavailable for session_search", exc_info=True)
            from hermes_state import format_session_db_unavailable
            return tool_error(format_session_db_unavailable(), success=False)

    # Mode normalisation. ``None`` / empty string / non-string → fall back to
    # the user's configured default (via ~/.hermes/config.yaml, see
    # ``_resolve_user_default_mode``). Defaults to "summary" if unset. We only
    # resolve the user default when the caller didn't pass an explicit mode —
    # an explicit "fast" or "summary" or "guided" wins regardless of config.
    if not isinstance(mode, str) or not mode.strip():
        mode = _resolve_user_default_mode()
    else:
        mode = mode.strip().lower()
    if mode in ("summarized", "summarise", "summarize", "deep"):
        mode = "summary"
    if mode in ("drill", "drilldown", "drill-down", "anchor", "around"):
        mode = "guided"
    if mode not in ("fast", "summary", "guided"):
        mode = "summary"

    # Guided mode is a different shape: it doesn't search, it drills. Branch
    # before FTS5 so we don't pay for anything we don't use, and so missing-arg
    # validation happens up front.
    if mode == "guided":
        return _guided_drill_down(
            db=db,
            session_id=session_id,
            around_message_id=around_message_id,
            window=window,
            current_session_id=current_session_id,
            anchors=anchors,
        )

    # Defensive: models (especially open-source) may send non-int limit values
    # (None when JSON null, string "int", or even a type object).  Coerce to a
    # safe integer before any arithmetic/comparison to prevent TypeError.
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 3
    limit = max(1, min(limit, 10))  # Clamp to [1, 10]

    # Recent sessions mode: when query is empty, return metadata for recent sessions.
    # No LLM calls — just DB queries for titles, previews, timestamps.
    if not query or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id)

    query = query.strip()

    try:
        # Parse role filter. When caller didn't pass one, default to
        # user+assistant — tool messages are usually noisy (serialised tool
        # calls, large outputs) and rarely the signal someone is searching
        # for. Callers can opt back in by passing role_filter='user,assistant,tool'
        # or just 'tool' when debugging tool output.
        role_list = None
        if role_filter and role_filter.strip():
            role_list = [r.strip() for r in role_filter.split(",") if r.strip()]
        else:
            role_list = ["user", "assistant"]

        # FTS5 search -- get matches ranked by relevance
        raw_results = db.search_messages(
            query=query,
            role_filter=role_list,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            limit=50,  # Get more matches to find unique sessions
            offset=0,
        )

        if not raw_results:
            return json.dumps({
                "success": True,
                "mode": mode,
                "query": query,
                "results": [],
                "count": 0,
                "message": "No matching sessions found.",
            }, ensure_ascii=False)

        # Resolve child sessions to their parent — delegation stores detailed
        # content in child sessions, but the user's conversation is the parent.
        def _resolve_to_parent(session_id: str) -> str:
            """Walk delegation chain to find the root parent session ID."""
            visited = set()
            sid = session_id
            while sid and sid not in visited:
                visited.add(sid)
                try:
                    session = db.get_session(sid)
                    if not session:
                        break
                    parent = session.get("parent_session_id")
                    if parent:
                        sid = parent
                    else:
                        break
                except Exception as e:
                    logging.debug(
                        "Error resolving parent for session %s: %s",
                        sid,
                        e,
                        exc_info=True,
                    )
                    break
            return sid

        current_lineage_root = (
            _resolve_to_parent(current_session_id) if current_session_id else None
        )

        # Group by resolved (parent) session_id, dedup, skip the current
        # session lineage. Compression and delegation create child sessions
        # that still belong to the same active conversation.
        #
        # IMPORTANT: we group BY parent (so the user sees one entry per
        # conversation lineage), but we preserve the raw FTS5 session_id on
        # the surviving result. The raw sid is the only sid that pairs
        # validly with ``match_message_id``; rewriting it to the parent
        # produces a "{parent_sid, child_message_id}" handle that guided
        # mode cannot resolve (#regression introduced by the original
        # match_message_id rollout). See the parent_session_id field in
        # fast-mode output for the lineage-root link the user expects to
        # see.
        seen_sessions = {}
        for result in raw_results:
            raw_sid = result["session_id"]
            resolved_sid = _resolve_to_parent(raw_sid)
            # Skip the current session lineage — the agent already has that
            # context, even if older turns live in parent fragments.
            if current_lineage_root and resolved_sid == current_lineage_root:
                continue
            if current_session_id and raw_sid == current_session_id:
                continue
            if resolved_sid not in seen_sessions:
                result = dict(result)
                # Keep raw_sid as session_id; expose lineage root separately.
                result["session_id"] = raw_sid
                if resolved_sid and resolved_sid != raw_sid:
                    result["parent_session_id"] = resolved_sid
                seen_sessions[resolved_sid] = result
            if len(seen_sessions) >= limit:
                break

        if mode == "fast":
            results = []
            for lineage_root, match_info in seen_sessions.items():
                # ``lineage_root`` is the dict key (resolved parent — used for
                # dedup grouping). ``match_info["session_id"]`` is the raw FTS5
                # row's session — the only sid that pairs with
                # ``match_info["id"]`` (the message id). Emit the pair (raw sid +
                # match_message_id) so the agent's follow-up
                # mode='guided' call has a valid {session_id, around_message_id}
                # handle. ``parent_session_id`` (if different) tells the agent
                # which conversation lineage this fragment belongs to.
                hit_sid = match_info.get("session_id") or lineage_root
                try:
                    session_meta = db.get_session(lineage_root) or {}
                except Exception:
                    session_meta = {}
                snippet = match_info.get("snippet") or ""
                context = match_info.get("context") or []
                if not isinstance(context, list):
                    context = []
                entry = {
                    "session_id": hit_sid,
                    "when": _format_timestamp(
                        session_meta.get("started_at") or match_info.get("session_started")
                    ),
                    "source": session_meta.get("source") or match_info.get("source", "unknown"),
                    "model": session_meta.get("model") or match_info.get("model") or "unknown",
                    "matched_role": match_info.get("role"),
                    "match_message_id": match_info.get("id"),
                    "title": session_meta.get("title") or None,
                    "snippet": snippet,
                    "context": context,
                    "summary": "[Search hit — summary not generated in fast mode] Use snippet/context fields, or set mode='summary' for LLM-generated recall.",
                }
                # Only emit parent_session_id when the FTS5 row lives in a
                # child of the displayed lineage — keeps the common case
                # (no delegation/compression) tidy.
                parent_sid = match_info.get("parent_session_id")
                if parent_sid and parent_sid != hit_sid:
                    entry["parent_session_id"] = parent_sid
                results.append(entry)

            return json.dumps({
                "success": True,
                "mode": "fast",
                "query": query,
                "results": results,
                "count": len(results),
                "sessions_searched": len(seen_sessions),
                "message": "Fast search returned FTS snippets without LLM summarization. Use mode='summary' for focused summaries when needed.",
            }, ensure_ascii=False)

        # Prepare all sessions for parallel summarization
        tasks = []
        for session_id, match_info in seen_sessions.items():
            try:
                messages = db.get_messages_as_conversation(session_id)
                if not messages:
                    continue
                session_meta = db.get_session(session_id) or {}
                conversation_text = _format_conversation(messages)
                conversation_text = _truncate_around_matches(conversation_text, query)
                tasks.append((session_id, match_info, conversation_text, session_meta))
            except Exception as e:
                logging.warning(
                    "Failed to prepare session %s: %s",
                    session_id,
                    e,
                    exc_info=True,
                )

        # Summarize all sessions in parallel
        async def _summarize_all() -> List[Union[tuple, Exception]]:
            """Summarize all sessions with bounded concurrency."""
            max_concurrency = min(_get_session_search_max_concurrency(), max(1, len(tasks)))
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _bounded_summary(text: str, meta: Dict[str, Any]):
                async with semaphore:
                    return await _summarize_session(text, query, meta)

            coros = [
                _bounded_summary(text, meta)
                for _, _, text, meta in tasks
            ]
            return await asyncio.gather(*coros, return_exceptions=True)

        try:
            # Use _run_async() which properly manages event loops across
            # CLI, gateway, and worker-thread contexts.  The previous
            # pattern (asyncio.run() in a ThreadPoolExecutor) created a
            # disposable event loop that conflicted with cached
            # AsyncOpenAI/httpx clients bound to a different loop,
            # causing deadlocks in gateway mode (#2681).
            from model_tools import _run_async
            results = _run_async(_summarize_all())
        except concurrent.futures.TimeoutError:
            logging.warning(
                "Session summarization timed out after 60 seconds",
                exc_info=True,
            )
            return json.dumps({
                "success": False,
                "error": "Session summarization timed out. Try a more specific query or reduce the limit.",
            }, ensure_ascii=False)

        summaries = []
        aux_total = {
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "call_count": 0,
        }
        for (session_id, match_info, conversation_text, session_meta), result in zip(tasks, results):
            usage: Optional[Dict[str, Any]] = None
            if isinstance(result, Exception):
                logging.warning(
                    "Failed to summarize session %s: %s",
                    session_id, result, exc_info=True,
                )
                summary_text = None
            elif isinstance(result, tuple):
                summary_text, usage = result
            else:
                # Defensive: a future code path might still return a bare string.
                summary_text, usage = result, None

            # Prefer resolved parent session metadata over FTS5 match metadata.
            # match_info carries source/model from the *child* session that contained
            # the FTS5 hit; after _resolve_to_parent() the session_id points to the
            # root, so session_meta has the authoritative platform/source for the
            # session the user actually cares about (#15909).
            entry = {
                "session_id": session_id,
                "when": _format_timestamp(
                    session_meta.get("started_at") or match_info.get("session_started")
                ),
                "source": session_meta.get("source") or match_info.get("source", "unknown"),
                "model": session_meta.get("model") or match_info.get("model"),
            }

            if summary_text:
                entry["summary"] = summary_text
            else:
                # Fallback: raw preview so matched sessions aren't silently
                # dropped when the summarizer is unavailable (fixes #3409).
                preview = (conversation_text[:500] + "\n…[truncated]") if conversation_text else "No preview available."
                entry["summary"] = f"[Raw preview — summarization unavailable]\n{preview}"

            if usage:
                entry["aux_usage"] = usage
                aux_total["model"] = aux_total["model"] or usage.get("model")
                aux_total["input_tokens"] += usage["input_tokens"]
                aux_total["output_tokens"] += usage["output_tokens"]
                aux_total["cache_read_tokens"] += usage["cache_read_tokens"]
                aux_total["cache_creation_tokens"] += usage["cache_creation_tokens"]
                aux_total["call_count"] += 1

            summaries.append(entry)

        payload = {
            "success": True,
            "mode": "summary",
            "query": query,
            "results": summaries,
            "count": len(summaries),
            "sessions_searched": len(seen_sessions),
        }
        # Only surface aux_usage_total when we actually captured any (test mocks
        # and providers that don't report usage produce an all-zero/empty dict —
        # don't pollute the payload in that case).
        if aux_total["call_count"]:
            payload["aux_usage_total"] = aux_total
        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        logging.error("Session search failed: %s", e, exc_info=True)
        return tool_error(f"Search failed: {str(e)}", success=False)


def check_session_search_requirements() -> bool:
    """Requires SQLite state database; summary mode also needs an auxiliary model."""
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search your long-term memory of past conversations, browse recent sessions, or drill "
        "into a specific session. This is your recall -- every past session is searchable.\n\n"
        "DEFAULT RECALL PATH: fast → guided.\n"
        "  • mode='fast' (default starting move) — FTS5 snippets + 1 message of context, no LLM "
        "call. ~10ms, ~1 KB per session. Use this for ANY recall question — discovery AND state "
        "reconstruction. Returns session_id + match_message_id anchors you then drill into.\n"
        "  • mode='guided' (standard follow-up) — given (session_id, message_id) anchors from a "
        "prior fast call, returns a window of raw messages around each anchor plus session "
        "bookends (first/last few user+assistant messages, when they don't already overlap the "
        "window). No LLM, no truncation, ~ms latency. This is how you actually read what "
        "happened — fast finds the sessions, guided reads the transcript with start-and-end "
        "context guaranteed. Tool messages around the anchor are filtered (anchor itself "
        "preserved) so payload stays signal-dense. **Never invent anchors or guess session_ids — "
        "guided will reject pairs that don't match real messages.**\n"
        "  • mode='summary' — LLM-generated recap across matched sessions. ~30s, ~$2/call in aux "
        "LLM cost. Trades latency + cost for prose synthesis. Reach for it when you genuinely "
        "need cross-session synthesis in one shot AND a fast→guided walk would be too many "
        "round-trips. Legitimate when the user explicitly asks for it or has configured it as "
        "their default — not legitimate as a reflexive choice when you haven't thought about "
        "the trade.\n\n"
        "PICKING A MODE WHEN UNSET: if the user hasn't configured a default and hasn't asked "
        "for a specific mode, prefer fast. Don't reflexively pick summary for 'catch me up' "
        "or 'what did we decide' questions — fast→guided answers those cheaper and shows you "
        "the actual messages instead of an LLM recap of them.\n\n"
        "MULTI-SESSION CATCH-UP: when a topic spans multiple sessions (e.g. 'where did we get to "
        "with X' over several days, or active work that's been touched in several recent "
        "sessions), do NOT drill only the top fast hit. Pass the top 2–3 hits as a multi-anchor "
        "guided call: ``mode='guided', anchors=[{session_id, around_message_id}, ...]``. Each "
        "anchor returns its own window + bookends in one call, so you see the full arc instead "
        "of one session's slice. A single-anchor drill is fine when the topic is contained to "
        "one session.\n\n"
        "WHEN FAST SNIPPETS LOOK NOISY: if fast's snippets all read like the same keywords "
        "echoing (because the searched topic IS the subject of those sessions — e.g. searching "
        "for 'session_search' in sessions about session_search), the snippets are decorative, "
        "not signal. The signal is the (session_id, match_message_id) pair. Do NOT pivot to "
        "filesystem/SQL/grep — that's the same shape failure as reflexive summary, just with "
        "manual archaeology instead of LLM telephone. Drill the top 2–3 hits with guided; "
        "bookend_end carries the session's prose resolution that snippets routinely miss.\n\n"
        "READING GUIDED RESPONSES: every guided window has three slices — ``bookend_start`` "
        "(opening prose of that session, may be empty when the window already covers the "
        "start), ``messages`` (the anchored window itself, the FTS5 hit + its neighbours), and "
        "``bookend_end`` (closing prose of that session). Read all three — the resolution lives "
        "in bookend_end, the goal in bookend_start. Skipping them leaves you anchored on the "
        "FTS5 hit alone and missing what came before/after.\n\n"
        "LINEAGE AWARENESS: sessions can be split by context compaction — when this happens, "
        "the child session's first messages are a post-compaction handoff, NOT the original "
        "arc opener. Spot it by ``parent_session_id`` on a fast hit (or in the guided response's "
        "session_meta). If ``bookend_start`` reads like a summary-of-prior-work rather than a "
        "user kickoff, the real opener is in the parent — fast-search again scoped to the "
        "parent if you need it. Most sessions are not compacted; this only matters on long "
        "multi-day arcs.\n\n"
        "Browsing recent sessions: call with NO arguments to see what was worked on recently. "
        "Returns titles, previews, timestamps. Zero LLM cost, instant. Start here when the user "
        "asks 'what were we working on' or 'what did we do recently'.\n\n"
        "The default mode is configurable per-user via ``auxiliary.session_search.default_mode`` "
        "in ~/.hermes/config.yaml (``fast`` | ``summary``). When no mode is passed explicitly, "
        "that user-configured value applies (then 'summary' as the final fallback for backward "
        "compatibility). Respect the configured default — if a user set ``summary``, they made "
        "that trade deliberately; if they set ``fast``, use fast without second-guessing.\n\n"
        "USE THIS PROACTIVELY when:\n"
        "- **BEFORE reaching for `gh`, GitHub API, web search, or file inspection**: if the user "
        "asks about the status of any project, branch, PR, design, or topic that's been worked on "
        "before, call session_search FIRST. The session DB carries what was DISCUSSED and DECIDED; "
        "external tools only show the current world state. Use session_search to find context, "
        "then external tools to verify reality.\n"
        "- The user says 'we did this before', 'remember when', 'last time', 'as I mentioned'\n"
        "- The user asks about a topic you worked on before but don't have in current context\n"
        "- The user references a project, person, or concept that seems familiar but isn't in memory\n"
        "- You want to check if you've solved a similar problem before\n"
        "- The user asks 'what did we do about X?' or 'how did we fix Y?'\n\n"
        "Don't hesitate to search when it is actually cross-session — fast mode is ~10ms and free. "
        "Better to search and confirm than to guess or ask the user to repeat themselves.\n\n"
        "Search syntax (modes 'fast' and 'summary'): keywords joined with OR for broad recall "
        "(elevenlabs OR baseten OR funding), phrases for exact match (\"docker networking\"), "
        "boolean (python NOT java), prefix (deploy*). "
        "IMPORTANT: Use OR between keywords for best results — FTS5 defaults to AND which misses "
        "sessions that only mention some terms. If a broad OR query returns nothing, try individual "
        "keyword searches in parallel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (modes 'fast' and 'summary'). Keywords, phrases, or boolean expressions to find in past sessions. Omit this parameter entirely to browse recent sessions instead. Ignored when mode='guided'.",
            },
            "role_filter": {
                "type": "string",
                "description": "Optional: only search messages from specific roles (comma-separated). Defaults to 'user,assistant' for fast/summary modes — tool messages are usually noisy (large outputs, serialised tool calls). Pass 'user,assistant,tool' to include tool output (debugging tool behaviour) or 'tool' to search tool output only. Ignored when mode='guided'.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to return (default: 3, max: 10). Bump higher (5–10) when the user wants to be in the retrieval loop and pick the right anchor for a guided drill-down. Ignored when mode='guided' (which returns one anchored window per anchor).",
                "default": 3,
            },
            "mode": {
                "type": "string",
                "enum": ["fast", "summary", "guided"],
                "description": (
                    "fast (default) — FTS5 snippets, no LLM, ~10ms. Use for any recall. "
                    "guided — REQUIRES anchors from a prior fast call; returns raw message "
                    "window around each anchor. summary — LLM recap, ~30s, ~$2/call; opt-in "
                    "cross-session synthesis (respect user config if they set it, but don't "
                    "reflexively pick it). If you want to drill but have no anchors yet, call "
                    "fast first and use its match_message_id values. Never invent anchors. "
                    "See the tool description for when to use which."
                ),
                "default": "fast",
            },
            "anchors": {
                "type": "array",
                "description": "Required for mode='guided'. List of {session_id, around_message_id} dicts to drill into. Copy session_id and match_message_id verbatim from prior fast-mode results — they pair as a single self-consistent handle. Do NOT substitute parent_session_id (shown for display context only; pairs incorrectly with match_message_id). One anchor is fine when the topic lives in a single session; for multi-session catch-up (topic touched across several recent sessions), pass the top 2–3 fast hits as separate anchors in ONE call — each gets its own window + bookends in the response's 'windows' array.",
                "items": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "around_message_id": {"type": "integer"},
                    },
                    "required": ["session_id", "around_message_id"],
                },
            },
            "window": {
                "type": "integer",
                "description": "Mode='guided' only. Number of messages to return on each side of each anchor (the anchor itself is always included). Shared across all anchors in a multi-anchor call. Clamped to [1, 20]. Default 5.",
                "default": 5,
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        mode=args.get("mode"),
        session_id=args.get("session_id"),
        around_message_id=args.get("around_message_id"),
        window=args.get("window", 5),
        anchors=args.get("anchors"),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id")),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)
