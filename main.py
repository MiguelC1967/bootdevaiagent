import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from call_function import available_functions, call_function
from prompts import system_prompt

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")

if api_key is None:
    raise RuntimeError("No API Key")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def main():
    parser = argparse.ArgumentParser(description="Chatbot")

    parser.add_argument(
        "user_prompt",
        type=str,
        help="User prompt",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]

    for _ in range(20):
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            tools=available_functions,
        )

        message = response.choices[0].message

        # Important: save the assistant's response
        messages.append(message)

        # No more tools requested = final response
        if not message.tool_calls:
            print(f"Final response:\n{message.content}")
            return

        # Execute each requested tool
        for tool_call in message.tool_calls:
            if tool_call.type != "function":
                continue

            result_message = call_function(
                tool_call,
                verbose=args.verbose,
            )

            if not result_message.get("content"):
                raise RuntimeError("Tool call returned an empty result")

            # Important: save the tool result
            messages.append(result_message)

    print("Maximum iterations reached without a final response.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()