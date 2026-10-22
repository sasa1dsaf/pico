"""Native tool-calling agent loop for Anthropic-compatible providers."""

from __future__ import annotations

import time

from .checkpoint import CHECKPOINT_NONE_STATUS
from .native_context import NATIVE_INPUT_BUDGET_TOKENS, compress_native_messages
from .providers.native import to_json_schema
from .task_state import TaskState
from .workspace import clip, now


def _recent_read_paths(agent, messages, limit=30):
    paths = set()
    for message in messages[-limit:]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") != "read_file":
                continue
            raw_path = (block.get("input") or {}).get("path")
            if not raw_path:
                continue
            try:
                paths.add(agent.memory.canonical_path(raw_path))
            except Exception:
                paths.add(str(raw_path))
    return paths


def run_native_loop(loop, user_message):
    agent = loop.agent
    run_started_at = time.monotonic()
    agent.memory.set_task_summary(user_message)

    task_state = TaskState.create(
        run_id=agent.new_run_id(),
        task_id=agent.new_task_id(),
        user_request=user_message,
    )
    agent.current_task_state = task_state
    agent.current_run_dir = agent.run_store.start_run(task_state)
    agent.emit_trace(
        task_state,
        "run_started",
        {
            "task_id": task_state.task_id,
            "user_request": clip(user_message, 300),
            "protocol": "native",
        },
    )

    stored_messages = list(agent.session.get("native_messages", []))
    base_messages, compression_metadata = compress_native_messages(
        stored_messages, max_tokens=int(NATIVE_INPUT_BUDGET_TOKENS * 0.85)
    )
    recent_paths = _recent_read_paths(agent, base_messages)
    sections = agent.context_manager.build_native(
        user_message, exclude_paths=recent_paths
    )
    task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
    prompt_metadata = dict(sections.metadata)
    prompt_metadata["history_compression"] = compression_metadata
    agent.last_prompt_metadata = prompt_metadata
    agent.emit_trace(
        task_state,
        "prompt_built",
        {
            "prompt_metadata": prompt_metadata,
            "purpose": "native_action",
        },
    )

    agent.record({"role": "user", "content": user_message, "created_at": now()})

    tool_schemas = to_json_schema(agent.tools)
    history_block = sections.history_text if not stored_messages else ""
    content = [
        {"type": "text", "text": f"<memory>\n{sections.memory_text}\n</memory>"},
    ]
    if history_block and history_block.strip() not in {
        "Transcript:\n- empty",
        "Transcript:\n- empty\n",
    }:
        content.append(
            {"type": "text", "text": f"<history>\n{history_block}\n</history>"}
        )
    content.append(
        {"type": "text", "text": f"Current user request:\n{user_message}"}
    )
    current_user_message = {"role": "user", "content": content}
    persist_user_message = {
        "role": "user",
        "content": [{"type": "text", "text": user_message}],
    }
    request_messages = [*base_messages, current_user_message]
    run_turns = []

    tool_steps = 0
    attempts = 0
    max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)

    while tool_steps < agent.max_steps and attempts < max_attempts:
        attempts += 1
        task_state.record_attempt()
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "model_requested",
            {
                "attempts": task_state.attempts,
                "tool_steps": task_state.tool_steps,
                "purpose": "native_action",
                "protocol": "native",
            },
        )
        agent.emit_progress(
            f"[pico] waiting for model (native, attempt {task_state.attempts}, step {task_state.tool_steps})..."
        )
        model_started_at = time.monotonic()
        try:
            turn = agent.model_client.complete_native(
                system=sections.system_text,
                messages=request_messages,
                tools=tool_schemas,
                max_new_tokens=agent.max_new_tokens,
                tool_choice={"type": "auto"},
            )
        except Exception as exc:
            completion_metadata = dict(
                getattr(agent.model_client, "last_completion_metadata", {}) or {}
            )
            if completion_metadata:
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            agent.session["native_messages"] = [
                *stored_messages,
                persist_user_message,
                *run_turns,
            ]
            agent.session_store.save(agent.session)
            loop._persist_model_failure(
                task_state,
                user_message,
                exc,
                run_started_at,
                prompt_metadata,
            )
            raise

        completion_metadata = dict(
            getattr(agent.model_client, "last_completion_metadata", {}) or {}
        )
        if completion_metadata:
            prompt_metadata.update(completion_metadata)
        agent.last_completion_metadata = completion_metadata
        agent.last_prompt_metadata = prompt_metadata
        duration_ms = int((time.monotonic() - model_started_at) * 1000)
        output_tokens = completion_metadata.get("output_tokens")
        token_suffix = f" ({output_tokens} tokens)" if output_tokens else ""
        agent.emit_progress(
            f"[pico] model responded in {duration_ms / 1000:.1f}s{token_suffix}"
        )
        agent.emit_trace(
            task_state,
            "model_parsed",
            {
                "kind": "native",
                "stop_reason": turn.stop_reason,
                "tool_calls": [
                    {"id": call.id, "name": call.name}
                    for call in turn.tool_calls
                ],
                "completion_metadata": completion_metadata,
                "duration_ms": duration_ms,
                "purpose": "native_action",
                "protocol": "native",
            },
        )

        if turn.tool_calls:
            tool_results = []
            for call in turn.tool_calls:
                tool_started_at = time.monotonic()
                tool_result = agent.execute_tool(call.name, call.args)
                result = tool_result.content
                tool_metadata = dict(tool_result.metadata or {})
                if tool_metadata.get("tool_status") != "rejected":
                    tool_steps += 1
                    task_state.record_tool(call.name)
                agent.record(
                    {
                        "role": "tool",
                        "name": call.name,
                        "args": call.args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": call.name,
                        "args": call.args,
                        "result": clip(result, 500),
                        "duration_ms": int(
                            (time.monotonic() - tool_started_at) * 1000
                        ),
                        **tool_metadata,
                    },
                )
                agent.emit_progress(f"[pico] tool {call.name}: {clip(result, 160)}")
                checkpoint = agent.create_checkpoint(
                    task_state, user_message, trigger="tool_executed"
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result,
                }
                if tool_metadata.get("tool_status") in {"rejected", "error"}:
                    result_block["is_error"] = True
                tool_results.append(result_block)

            assistant_message = {"role": "assistant", "content": turn.raw_content}
            tool_message = {"role": "user", "content": tool_results}
            request_messages.extend([assistant_message, tool_message])
            run_turns.extend([assistant_message, tool_message])
            continue

        if turn.stop_reason == "max_tokens":
            if turn.raw_content:
                assistant_message = {"role": "assistant", "content": turn.raw_content}
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "(response truncated)"}],
                }
            notice_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Runtime notice: previous response was truncated at the "
                            "max output token limit. Do not explain; reply with text "
                            "or a tool call."
                        ),
                    }
                ],
            }
            request_messages.extend([assistant_message, notice_message])
            run_turns.extend([assistant_message, notice_message])
            agent.emit_progress(
                f"[pico] retry {attempts}/{max_attempts}: native output truncated; retrying..."
            )
            continue

        if turn.text.strip():
            final = turn.text.strip()
            if final.startswith("<final>") and final.endswith("</final>"):
                final = final[len("<final>") : -len("</final>")].strip()
            final_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": final}],
            }
            agent.session["native_messages"] = [
                *stored_messages,
                persist_user_message,
                *run_turns,
                final_message,
            ]
            agent.session_store.save(agent.session)
            return loop._finish_success(
                task_state, user_message, final, run_started_at
            )

        if turn.stop_reason == "tool_use":
            notice = (
                "Runtime notice: the previous response stopped for tool_use but "
                "contained no valid tool call. Reply with exactly one valid tool call."
            )
        else:
            notice = (
                "Runtime notice: empty native response. Reply with text or a tool call."
            )
        if turn.raw_content:
            assistant_message = {"role": "assistant", "content": turn.raw_content}
        else:
            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": "(empty response)"}],
            }
        notice_message = {
            "role": "user",
            "content": [{"type": "text", "text": notice}],
        }
        request_messages.extend([assistant_message, notice_message])
        run_turns.extend([assistant_message, notice_message])
        agent.emit_progress(
            f"[pico] retry {attempts}/{max_attempts}: malformed native response; retrying..."
        )

    if attempts >= max_attempts and tool_steps < agent.max_steps:
        final = "Stopped after too many malformed native responses without a final answer."
        task_state.stop_retry_limit(final)
    else:
        final = "Stopped after reaching the step limit without a final answer."
        task_state.stop_step_limit(final)
    final_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": final}],
    }
    agent.session["native_messages"] = [
        *stored_messages,
        persist_user_message,
        *run_turns,
        final_message,
    ]
    agent.session_store.save(agent.session)
    agent.record({"role": "assistant", "content": final, "created_at": now()})
    if not agent.has_durable_items():
        agent.promote_durable_memory(user_message, final)
    agent.run_store.write_task_state(task_state)
    checkpoint = agent.create_checkpoint(
        task_state, user_message, trigger=task_state.stop_reason or "run_stopped"
    )
    agent.emit_trace(
        task_state,
        "checkpoint_created",
        {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "trigger": task_state.stop_reason or "run_stopped",
        },
    )
    agent.emit_trace(
        task_state,
        "run_finished",
        {
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": final,
            "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            "protocol": "native",
        },
    )
    agent.run_store.write_report(
        task_state, agent.redact_artifact(agent.build_report(task_state))
    )
    return final
