"""
Transportation & Assignment AI Agent (Google Gemini version)
============================================================
Describe a problem in plain English (or paste a matrix) and the agent will:
  1. understand it and pull out the numbers,
  2. choose the right tool (transportation or assignment),
  3. run the Python solvers in solvers.py,
  4. explain the answer in simple words.

Run:
    python agent.py            -> start the chat
    python agent.py --models   -> list Gemini models available to your key
"""

import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

import solvers

# ---------------------------------------------------------------
# 0. SETTINGS  (the key is read from a .env file - never hard-code it!)
# ---------------------------------------------------------------
load_dotenv()                                            # loads variables from .env
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")    # change in .env if needed
MAX_TOOL_ROUNDS = 8                                      # safety limit for the agent loop

# ---------------------------------------------------------------
# 1. THE AGENT'S INSTRUCTIONS (system prompt)
# ---------------------------------------------------------------
SYSTEM_PROMPT = """You are an Operations Research assistant that helps students solve
Transportation and Assignment problems.

How you work:
- The user may describe a problem in words, paste tables, or give raw matrices.
  Extract the numbers carefully and use the tools to solve it. Never calculate the
  answer yourself - always use a tool.
- Transportation problem: needs a cost matrix (rows = sources, columns = destinations),
  a supply list and a demand list.
- Assignment problem: needs a matrix (rows = workers/machines, columns = jobs).
  Decide if the goal is minimise (cost, time, distance) or maximise (profit, score).
- Before calling a tool, briefly restate the data you extracted (matrix, supply, demand)
  so the user can spot any mistake.
- If any needed number is missing or unclear, ask ONE short question instead of guessing.
- For transportation, if the user names a method (North-West Corner, Least Cost,
  Vogel/VAM, or optimal) use that. If they do not name one, use method "all" so they can
  compare methods.
- After getting results, explain them simply: show who ships/gets what, the total cost,
  and (for transportation) how the heuristic compares to the optimal cost.
  Mention if a dummy row/column was added and why.
- If a tool returns an error, fix the input if you can, otherwise tell the user what is wrong.
- If the question is unrelated to these two problem types, say so politely.
- Keep answers clear and short. Use small tables when helpful."""

# ---------------------------------------------------------------
# 2. THE TOOLS (what the agent is allowed to do)
# ---------------------------------------------------------------
transportation_tool = types.FunctionDeclaration(
    name="solve_transportation",
    description=(
        "Solve a transportation problem. Handles unbalanced problems automatically "
        "by adding a dummy source/destination. Returns allocations and total cost."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cost": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Cost matrix. cost[i][j] = cost per unit from source i to destination j.",
            },
            "supply": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Supply (capacity) of each source, in row order.",
            },
            "demand": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Demand of each destination, in column order.",
            },
            "method": {
                "type": "string",
                "enum": ["all", "northwest", "least_cost", "vogel", "optimal"],
                "description": "Which method to use. 'all' runs every method for comparison.",
            },
        },
        "required": ["cost", "supply", "demand"],
    },
)

assignment_tool = types.FunctionDeclaration(
    name="solve_assignment",
    description=(
        "Solve an assignment problem with the Hungarian method. Works for square and "
        "rectangular matrices. Returns the best one-to-one assignment and the total."
    ),
    parameters={
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Matrix of costs or profits. Rows = workers, columns = jobs.",
            },
            "objective": {
                "type": "string",
                "enum": ["min", "max"],
                "description": "'min' to minimise cost/time, 'max' to maximise profit.",
            },
        },
        "required": ["matrix", "objective"],
    },
)

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=[transportation_tool, assignment_tool])],
    # We run the tool loop ourselves so you can see every step of the agent.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


# ---------------------------------------------------------------
# 3. RUNNING A TOOL
# ---------------------------------------------------------------
def to_plain(value):
    """Gemini sends numbers/lists in its own wrappers; turn them into plain Python."""
    return json.loads(json.dumps(value, default=list))


def run_tool(name, args):
    """Call the matching Python solver. Returns a dict (either a result or an error)."""
    try:
        args = to_plain(dict(args))
        if name == "solve_transportation":
            result = solvers.solve_transportation(
                args["cost"], args["supply"], args["demand"], args.get("method", "all")
            )
        elif name == "solve_assignment":
            result = solvers.solve_assignment(args["matrix"], args["objective"])
        else:
            return {"error": f"Unknown tool: {name}"}
        return {"result": result}
    except Exception as e:                  # send the error back so the model can react
        return {"error": str(e)}


# ---------------------------------------------------------------
# 4. ONE TURN OF THE AGENT (the "agent loop")
# ---------------------------------------------------------------
def agent_turn(client, contents):
    """Send the chat to Gemini. Keep running tools until Gemini gives a final answer."""
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=CONFIG
        )

        if not response.candidates or response.candidates[0].content is None:
            return "(The model returned no answer - it may have been blocked. Try rephrasing.)"

        # Remember what the model said (keep it exactly as returned, incl. tool requests)
        contents.append(response.candidates[0].content)

        calls = response.function_calls or []
        if not calls:                        # no tool needed -> this is the final answer
            return response.text or ""

        # The model wants tools -> run each one and send the results back
        result_parts = []
        for call in calls:
            print(f"   [agent is calling: {call.name}]")
            outcome = run_tool(call.name, call.args or {})
            result_parts.append(
                types.Part.from_function_response(name=call.name, response=outcome)
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return "(Stopped: too many tool steps in one turn. Please try again.)"


# ---------------------------------------------------------------
# 5. HELPERS + CHAT LOOP
# ---------------------------------------------------------------
def explain_api_error(e):
    """Turn Gemini API errors into friendly hints."""
    if e.code == 429:
        return ("Rate limit or daily free quota reached. Wait a minute and retry, or set a "
                "lighter model in .env (e.g. GEMINI_MODEL=gemini-3.5-flash-lite).")
    if e.code == 404:
        return (f"Model '{MODEL}' was not found. Run `python agent.py --models` to see "
                "available models, then set GEMINI_MODEL in your .env file.")
    if e.code in (400, 401, 403):
        return f"Request rejected ({e.code}). Check that your API key is valid. Details: {e.message}"
    return f"Gemini API error {e.code}: {e.message}"


def list_models(client):
    print("Gemini models available to your key:\n")
    for m in client.models.list():
        if "gemini" in m.name:
            print("  ", m.name.replace("models/", ""))


def main():
    if not API_KEY:
        print("No API key found.\n"
              "1) Create a file named .env in this folder\n"
              "2) Add this line:  GEMINI_API_KEY=your-key-here\n"
              "Get a free key at https://aistudio.google.com/apikey")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    if "--models" in sys.argv:
        list_models(client)
        return

    contents = []                            # the full conversation (the agent's memory)

    print("=" * 60)
    print("  Transportation & Assignment AI Agent  (Gemini)")
    print(f"  Model: {MODEL}")
    print("  Describe your problem or paste your matrix. Type 'quit' to exit.")
    print("=" * 60)
    print("Example: 3 factories with supply 20, 30, 25 send goods to 3 shops")
    print("with demand 10, 35, 30. Costs: row1 = 8 6 10, row2 = 9 12 13, row3 = 14 9 16.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        checkpoint = len(contents)           # so we can undo this turn if it fails
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
        try:
            print("\nAgent:", agent_turn(client, contents), "\n")
        except errors.APIError as e:
            print(f"\n{explain_api_error(e)}\n")
            del contents[checkpoint:]


if __name__ == "__main__":
    main()