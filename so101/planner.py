"""The top layer: decides the next command.

A planner is asked for the next command whenever the layers below hand control back. It
gets the instruction, the commands the arm was trained on, what has happened so far, and
the current camera pictures. It answers with one of those commands, or "done", or "stop".

`ClaudePlanner` calls the Claude API. `ManualPlanner` asks at the keyboard. `ListPlanner`
follows a fixed list without looking.
"""
import base64
import io
import json
import threading
import time
from functools import partial

from PIL import Image

DONE, STOP = "done", "stop"

SYSTEM = """You are the planner for a small robot arm (an SO-101: five joints and a two-finger gripper) working on a tabletop. A person has given the arm an instruction. Your job is to choose the arm's next step.

How the arm works: below you is a learned model that can carry out a fixed set of short commands, listed in each request. It was trained on demonstrations of exactly those commands and understands nothing else, so the command you choose has to be copied from the list character for character. Each command runs until one of three things happens, and then you are asked again:

- "finished": the model believes it completed the command. It can be wrong. It only knows that the motion ended the way its demonstrations ended. It does not check which object it is holding or where an object ended up.
- "surprise": the gripper was holding something and suddenly felt nothing, so the object was probably dropped or knocked out.
- "timeout": the command ran out of time without finishing.

The model below cannot see whether the task as a whole is going well. You can. You are given the current camera pictures, and they are the ground truth: trust them over the event text and over your own expectation of what should have happened. Before choosing, look at where each object actually is and what, if anything, is in the gripper.

Choose "done" when the pictures show the instruction has been carried out. Choose "stop" if the instruction cannot be carried out with the available commands, if the scene looks unsafe (for example a hand in the workspace), or if the same step has failed several times in a row and repeating it is not helping. Otherwise choose the single command that makes the most progress from the scene as it is now, which after a failure is often a repeat of an earlier command.

Reply with a short description of what you see that bears on the choice, then the command."""


def to_jpeg(image, longest):
    """A camera frame (H, W, 3 uint8) as base64 JPEG, shrunk so its longest side is at most `longest`."""
    im = Image.fromarray(image)
    if max(im.size) > longest:
        im.thumbnail((longest, longest))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def describe(instruction, commands, history):
    lines = [f"Instruction from the person: {instruction}", "", "Commands the arm can carry out:"]
    lines += [f"- {c}" for c in commands]
    lines += ["", "What has happened so far:"]
    lines += [f"{i + 1}. command \"{c}\" ended with: {outcome}" for i, (c, outcome) in enumerate(history)] or ["Nothing yet. This is the first step."]
    lines += ["", "The pictures above are the cameras right now. What should the arm do next?"]
    return "\n".join(lines)


class PlannerUnavailable(RuntimeError):
    """The planner could not be reached or declined. The arm should hold still."""


class ClaudePlanner:
    def __init__(self, cfg, client=None):
        self.cfg = cfg
        if client is None:
            import anthropic
            # Reads ANTHROPIC_API_KEY from the environment. The SDK's own retries are off, because each
            # would get the full timeout again; `next` retries inside one overall time limit instead.
            client = anthropic.Anthropic(max_retries=0)
        self.client = client
        self.sleep = time.sleep
        self.last_observation = ""

    def next(self, instruction, commands, history, images):
        c = self.cfg
        content = []
        for name, image in images.items():
            content.append({"type": "text", "text": f"Camera: {name}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": to_jpeg(image, c.planner_image)}})
        content.append({"type": "text", "text": describe(instruction, commands, history)})
        schema = {"type": "object", "additionalProperties": False, "required": ["observation", "command"],
                  "properties": {"observation": {"type": "string"},
                                 "command": {"type": "string", "enum": list(commands) + [DONE, STOP]}}}
        allowed = list(commands) + [DONE, STOP]
        request = dict(model=c.planner_model, max_tokens=16000, system=SYSTEM,
                       messages=[{"role": "user", "content": content}],
                       output_config={"effort": c.planner_effort, "format": {"type": "json_schema", "schema": schema}})
        deadline = time.monotonic() + c.planner_timeout       # for the whole call, retries included
        attempt = 0
        while True:
            try:
                left = max(deadline - time.monotonic(), 0.1)
                if c.planner_fallbacks:
                    # If the model declines a request, the API re-runs it on a fallback model in the same call.
                    # `fallbacks` goes in `extra_body` because SDK releases up to at least 0.86 do not take
                    # it as an argument; passed directly it raises TypeError before anything is sent.
                    send = partial(self.client.beta.messages.create, betas=["server-side-fallback-2026-07-01"],
                                   extra_body={"fallbacks": "default"}, timeout=left, **request)
                else:
                    send = partial(self.client.messages.create, timeout=left, **request)
                # The SDK applies `timeout` to each stage of the request (connecting, each read), not to the
                # whole of it, so an answer that trickles in could run past the limit. `within` holds the
                # whole call to the wall clock.
                reply = within(left, send)
                break
            except Exception as e:             # whatever went wrong with the request, the arm holds still
                attempt += 1
                pause = min(2.0 ** (attempt - 1), 8.0)
                if not passing(e) or attempt > RETRIES or time.monotonic() + pause >= deadline:
                    raise PlannerUnavailable(explain(e)) from e
                self.sleep(pause)
        if reply.stop_reason == "refusal":
            raise PlannerUnavailable("the planner declined the request")
        if reply.stop_reason == "max_tokens":
            raise PlannerUnavailable("the planner's reply was cut off")
        try:
            # After a fallback the reply can hold text from the model that declined, then a "fallback"
            # block, then the fallback model's answer. The answer is the last text block.
            answer = json.loads([b.text for b in reply.content if b.type == "text"][-1])
            command, observation = answer["command"], str(answer.get("observation", ""))
        except (IndexError, ValueError, KeyError, TypeError, AttributeError) as e:
            raise PlannerUnavailable("the planner's reply could not be read") from e
        if command not in allowed:
            raise PlannerUnavailable(f"the planner chose a command the arm does not have: {command!r}")
        self.last_observation = observation
        return command


RETRIES = 2


def within(seconds, call):
    """The result of `call()`, or TimeoutError if it has not returned after `seconds` by the wall clock.

    The call runs on another thread. One that is given up on is left to end by itself and its answer
    is dropped; the thread does not keep the program alive.
    """
    box = {}

    def work():
        try:
            box["value"] = call()
        except Exception as e:           # raised below, on the caller's thread
            box["error"] = e
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"no answer after {seconds:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def passing(error):
    """Whether a failed request is worth sending again: the network, a rate limit, or a fault at the API."""
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    status = status if isinstance(status, int) else 0
    return name in ("APIConnectionError", "RateLimitError", "InternalServerError") or status in (408, 409, 429) or status >= 500


def explain(error):
    """A short reason for a failed API request, from the SDK's error class."""
    name = type(error).__name__
    if name == "AuthenticationError":
        return "the API key was rejected; set ANTHROPIC_API_KEY"
    if name == "RateLimitError":
        return "rate limited by the API after retries"
    if name in ("APITimeoutError", "TimeoutError"):
        return "the API took too long to answer"
    if name == "APIConnectionError":
        return "could not reach the API"
    if hasattr(error, "status_code"):
        return f"the API returned an error ({error.status_code}): {getattr(error, 'message', error)}"
    return f"the request failed ({name}): {error}"


class ManualPlanner:
    """A person types the command. Useful for testing the lower layers without an API key."""

    def __init__(self, ask=input, say=print):
        self.ask, self.say = ask, say
        self.last_observation = ""

    def next(self, instruction, commands, history, images):
        self.say(f"\nInstruction: {instruction}")
        if history:
            self.say(f"Last command \"{history[-1][0]}\" ended with: {history[-1][1]}")
        for i, c in enumerate(commands):
            self.say(f"  {i + 1}. {c}")
        while True:
            typed = self.ask("Number or command (or done / stop): ").strip()
            if typed.isdigit() and 1 <= int(typed) <= len(commands):
                return commands[int(typed) - 1]
            if typed in commands or typed in (DONE, STOP):
                return typed
            self.say("Not one of the commands.")


class ListPlanner:
    """Works through a fixed list. Moves on only when a command reports finished. Never looks."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.last_observation = ""

    def next(self, instruction, commands, history, images):
        done = sum(1 for _, outcome in history if outcome == "finished")
        return self.steps[done] if done < len(self.steps) else DONE
