"""Brain module for LLM-based reasoning and action selection."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Literal
from openai import OpenAI

# Setup logging - DEBUG to file, WARNING to console
file_handler = logging.FileHandler('agent.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)
# Reduce noise from HTTP libraries
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('uiautomator2').setLevel(logging.WARNING)

logger = logging.getLogger('brain')


@dataclass
class AgentAction:
    """Represents an action the agent should take."""
    action: Literal["click", "type", "scroll", "open_app", "back", "home", "done", "wait", "respond", "ask"]
    target_uid: Optional[int] = None
    text: Optional[str] = None
    app_package: Optional[str] = None
    direction: Optional[str] = None  # up, down, left, right
    message: Optional[str] = None  # For respond/ask actions
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target_uid": self.target_uid,
            "text": self.text,
            "app_package": self.app_package,
            "direction": self.direction,
            "message": self.message,
            "reasoning": self.reasoning,
        }


SYSTEM_PROMPT = """You are an Android automation agent. You control a phone and complete tasks step-by-step.

## CRITICAL: Response Format
You MUST respond with ONLY valid JSON. No thinking out loud. No explanations outside JSON.

{"action": "ACTION_TYPE", "target_uid": UID_OR_NULL, "text": "TEXT_OR_NULL", "app_package": "PACKAGE_OR_NULL", "direction": "DIRECTION_OR_NULL", "message": "MESSAGE_OR_NULL", "reasoning": "brief reason"}

## Actions
- `click`: Tap element by target_uid
- `type`: Enter text (use after clicking a text field)
- `scroll`: Scroll screen (direction: up/down/left/right)
- `open_app`: Launch app by package name
- `back`/`home`: Navigation buttons
- `wait`: Wait for UI to load
- `respond`: Answer a question (use message field)
- `ask`: Ask user for clarification (use message field)
- `done`: Task fully complete

## Key Rules

1. **Multi-step tasks**: Most tasks require MULTIPLE actions. After each action, analyze the NEW screen and continue. Only use `done` when the ENTIRE task is complete.

2. **Click strategically**: Look at the UI elements list. Find buttons/links that progress toward the goal. Use the target_uid.

3. **Don't give up early**: If you clicked something, wait for the result in the next step. Keep going until the task is done.

4. **App launches**: Use `open_app` with package name - never try to find app icons.

5. **Questions vs Actions**: 
   - If user asks a QUESTION (how many, what, which) → use `respond`
   - If user wants you to DO something → take actions until complete

## Examples

Task: "Book a court for tomorrow at 6pm"
Step 1: {"action": "click", "target_uid": 45, "reasoning": "Clicking date picker to select tomorrow"}
Step 2: {"action": "click", "target_uid": 67, "reasoning": "Selected tomorrow's date"}  
Step 3: {"action": "click", "target_uid": 89, "reasoning": "Clicking 6pm time slot"}
Step 4: {"action": "click", "target_uid": 102, "reasoning": "Clicking Book/Confirm button"}
Step 5: {"action": "done", "reasoning": "Booking confirmed"}

Task: "How many music apps do I have?"
Step 1: {"action": "respond", "message": "You have 3 music apps: Spotify, YouTube Music, and Apple Music.", "reasoning": "Answered question from app list"}

Task: "Open Spotify"
Step 1: {"action": "open_app", "app_package": "com.spotify.music", "reasoning": "Launching Spotify directly"}
"""


class AgentBrain:
    """LLM-based reasoning module for the Android agent."""
    
    def __init__(self, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        """Initialize the brain with OpenAI client."""
        self.model = model
        
        # Support local LLM servers
        if base_url:
            self.client = OpenAI(base_url=base_url, api_key="local")
        else:
            self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        
        self.conversation_history = []
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
    
    def think(
        self,
        user_goal: str,
        ui_summary: str,
        app_context: Optional[str] = None,
        previous_action: Optional[str] = None,
    ) -> AgentAction:
        """Determine the next action based on current state.
        
        Args:
            user_goal: What the user wants to accomplish
            ui_summary: Current UI state summary
            app_context: Optional context about relevant installed apps
            previous_action: What the agent did in the last step
        
        Returns:
            AgentAction describing what to do next
        """
        # Build the user message
        message_parts = [f"## User Goal\n{user_goal}"]
        
        if app_context:
            message_parts.append(f"## Relevant Installed Apps\n{app_context}")
        
        message_parts.append(f"## Current Screen\n{ui_summary}")
        
        if previous_action:
            message_parts.append(f"## Previous Action\n{previous_action}")
        
        user_message = "\n\n".join(message_parts)
        
        logger.debug(f"User message length: {len(user_message)} chars")
        
        # Add to conversation
        if not self.conversation_history:
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        logger.debug(f"Sending {len(self.conversation_history)} messages to LLM")
        
        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.1,
                max_tokens=2000,  # Increased for reasoning models
            )
            
            logger.debug(f"Raw API response: finish_reason={response.choices[0].finish_reason}")
            
            # Get the message content - some models put response in 'reasoning' field
            message = response.choices[0].message
            assistant_message = message.content
            
            # Check for reasoning field if content is empty (some Ollama models do this)
            if not assistant_message and hasattr(message, 'reasoning') and message.reasoning:
                logger.debug(f"Content empty, checking reasoning field")
                # Try to extract JSON from the reasoning
                reasoning = message.reasoning
                # Look for JSON in reasoning
                if '{' in reasoning and '}' in reasoning:
                    start = reasoning.find('{')
                    end = reasoning.rfind('}') + 1
                    assistant_message = reasoning[start:end]
                    logger.debug(f"Extracted JSON from reasoning: {assistant_message[:200]}")
                else:
                    # Just use the reasoning as a response
                    assistant_message = reasoning
            
            logger.debug(f"LLM response content: {repr(assistant_message[:200] if assistant_message else 'EMPTY')}")
            
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Parse the response
            action = self._parse_response(assistant_message)
            
            # If parsing failed (returned wait), add a correction message
            if action.action == "wait" and "retrying" in action.reasoning:
                self.conversation_history.append({
                    "role": "user", 
                    "content": "Please respond with ONLY valid JSON. No other text."
                })
            
            return action
            
        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            return AgentAction(
                action="done",
                reasoning=f"Error communicating with LLM: {e}"
            )
    
    def _parse_response(self, response: str) -> AgentAction:
        """Parse LLM response into an AgentAction."""
        logger.debug(f"Parsing response: {repr(response)}")
        
        if not response:
            logger.error("Empty response from LLM")
            return AgentAction(
                action="respond",
                message="I received an empty response. Could you please try again?",
                reasoning="LLM returned empty response"
            )
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                response = "\n".join(json_lines)
            
            # Try to find JSON in the response if it's mixed with text
            if not response.startswith("{"):
                # Look for JSON object in the response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    response = response[start:end]
                    logger.debug(f"Extracted JSON: {response}")
            
            data = json.loads(response)
            logger.debug(f"Parsed action: {data.get('action')}")
            
            return AgentAction(
                action=data.get("action", "done"),
                target_uid=data.get("target_uid"),
                text=data.get("text"),
                app_package=data.get("app_package"),
                direction=data.get("direction"),
                message=data.get("message"),
                reasoning=data.get("reasoning", ""),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {repr(response[:200])}")
            # If LLM returned plain text, it's likely thinking out loud - ask it to try again
            # Return a wait action to retry on next step
            return AgentAction(
                action="wait",
                reasoning=f"LLM returned non-JSON response, retrying"
            )
    
    def think_with_vision(
        self,
        user_goal: str,
        screenshot_base64: str,
        app_context: Optional[str] = None,
    ) -> AgentAction:
        """Use vision model for complex screens (fallback mode).
        
        This is slower but can understand visual elements that
        aren't well-represented in the XML hierarchy.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"User Goal: {user_goal}\n\nAnalyze this screenshot and determine the next action:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_base64}"
                        }
                    }
                ]
            }
        ]
        
        if app_context:
            messages[1]["content"][0]["text"] += f"\n\nRelevant Apps:\n{app_context}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Vision model
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )
            
            return self._parse_response(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Vision LLM error: {e}")
            return AgentAction(
                action="done",
                reasoning=f"Vision error: {e}"
            )
