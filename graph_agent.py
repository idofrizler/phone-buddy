"""LangGraph-based Android Agent with ReACT loop, persistence, and human-in-the-loop."""

import json
import logging
import operator
from typing import Annotated, Literal, Optional, TypedDict, Sequence
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Local imports
from connection_manager import ConnectionManager
from app_library import AppLibrary
from vision import VisionModule
from executor import ActionExecutor

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('agent.log')]
)
logger = logging.getLogger('graph_agent')


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State that flows through the graph."""
    # Core state
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_goal: str
    
    # Perception state
    current_screen: str
    current_app: str
    app_context: str
    
    # Execution state  
    step_count: int
    last_action: Optional[str]
    last_result: Optional[str]
    
    # Control flow
    status: Literal["running", "completed", "needs_input", "error"]
    pending_confirmation: Optional[str]  # Action waiting for user approval


# ============================================================================
# TOOLS - Actions the agent can take
# ============================================================================

# These will be bound to the actual device at runtime
_device_connection: Optional[ConnectionManager] = None
_app_library: Optional[AppLibrary] = None
_vision: Optional[VisionModule] = None
_executor: Optional[ActionExecutor] = None


@tool
def click_element(uid: int) -> str:
    """Click on a UI element by its UID number."""
    if not _executor:
        return "Error: Device not connected"
    success, msg = _executor._execute_click(uid)
    return msg


@tool
def type_text(text: str) -> str:
    """Type text into the currently focused field."""
    if not _executor:
        return "Error: Device not connected"
    success, msg = _executor._execute_type(text)
    return msg


@tool
def scroll_screen(direction: Literal["up", "down", "left", "right"]) -> str:
    """Scroll the screen in a direction."""
    if not _executor:
        return "Error: Device not connected"
    success, msg = _executor._execute_scroll(direction)
    return msg


@tool
def open_app(package_name: str) -> str:
    """Open an app by its package name (e.g., com.spotify.music)."""
    if not _executor:
        return "Error: Device not connected"
    success, msg = _executor._execute_open_app(package_name)
    return msg


@tool
def press_back() -> str:
    """Press the back button."""
    if not _executor:
        return "Error: Device not connected"
    success, msg = _executor._execute_back()
    return msg


@tool
def press_home() -> str:
    """Press the home button."""
    if not _executor:
        return "Error: Device not connected"
    success, msg = _executor._execute_home()
    return msg


@tool  
def wait_for_screen(seconds: int = 2) -> str:
    """Wait for the screen to update."""
    if not _executor:
        return "Error: Device not connected"
    import time
    time.sleep(seconds)
    return f"Waited {seconds} seconds"


@tool
def get_screen_state() -> str:
    """Get the current screen UI elements. Use this to see what's on screen."""
    if not _vision:
        return "Error: Vision module not connected"
    return _vision.get_ui_summary()


@tool
def search_installed_apps(query: str) -> str:
    """Search for installed apps by name. Returns matching app names and package IDs."""
    if not _app_library:
        return "Error: App library not initialized"
    matches = _app_library.fuzzy_find_app(query, threshold=60)
    if not matches:
        return f"No apps found matching '{query}'"
    lines = [f"Apps matching '{query}':"]
    for app in matches[:10]:
        lines.append(f"  - {app.common_name}: {app.package_name}")
    return "\n".join(lines)


@tool
def list_all_apps() -> str:
    """List all installed apps on the device."""
    if not _app_library:
        return "Error: App library not initialized"
    return _app_library.get_apps_summary(max_apps=200)


# All available tools
TOOLS = [
    click_element,
    type_text, 
    scroll_screen,
    open_app,
    press_back,
    press_home,
    wait_for_screen,
    get_screen_state,
    search_installed_apps,
    list_all_apps,
]


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are an Android automation agent. You control a phone and help users accomplish tasks.

## How You Work
You operate in a ReACT loop: Reason about the situation, then Act using tools, then Observe the result.

## Available Tools
- `click_element(uid)`: Tap a UI element by its UID
- `type_text(text)`: Type text into focused field
- `scroll_screen(direction)`: Scroll up/down/left/right
- `open_app(package_name)`: Launch an app by package name
- `press_back()`: Press back button
- `press_home()`: Press home button  
- `wait_for_screen(seconds)`: Wait for UI to update
- `get_screen_state()`: See current screen elements
- `search_installed_apps(query)`: Find apps by name
- `list_all_apps()`: See all installed apps

## Guidelines

1. **Start by understanding**: If you need info about apps or the screen, use the appropriate tools first.

2. **Multi-step tasks**: Most tasks require multiple actions. Keep going until the task is fully complete.

3. **Use tools strategically**: 
   - To open an app → use `open_app` with the package name, then wait 2 seconds for it to load
   - To find what's on screen → use `get_screen_state`
   - To click something → find its UID from screen state, then `click_element`

4. **When to STOP and respond**:
   - After successfully opening an app → respond with "Opened [app name]"
   - After completing a requested action → respond with confirmation
   - If task cannot be completed → explain why
   - If you need more info from user → ask a clarifying question

5. **Questions vs Actions**:
   - If user asks a question (how many, what, list) → gather info with tools, then respond
   - If user wants you to DO something → take action(s), then respond with result

IMPORTANT: After completing a task, respond with a short confirmation message. Don't keep trying if you already succeeded.
"""


# ============================================================================
# GRAPH NODES
# ============================================================================

def create_agent_graph(
    model_name: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    confirm_actions: bool = False,
):
    """Create the LangGraph agent.
    
    Args:
        model_name: The LLM model to use
        base_url: Optional base URL for local LLM (e.g., Ollama)
        confirm_actions: If True, pause before executing device actions
    """
    
    # Initialize LLM
    if base_url:
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key="local",
            temperature=0.1,
            max_tokens=2000,
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=2000,
        )
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # ---- Node: Perceive ----
    def perceive(state: AgentState) -> dict:
        """Gather current screen state and context."""
        import time
        logger.debug("Perceive node executing")
        
        # Wait a bit for screen to update after previous action
        if state.get("last_action"):
            time.sleep(1)
        
        updates = {
            "step_count": state.get("step_count", 0) + 1,
        }
        
        # Get current screen if we have vision module
        if _vision:
            try:
                screen = _vision.get_ui_summary()
                updates["current_screen"] = screen
                # Extract current app from first line
                if screen:
                    first_line = screen.split('\n')[0]
                    if "Current App:" in first_line:
                        updates["current_app"] = first_line.split("Current App:")[1].strip()
            except Exception as e:
                logger.error(f"Failed to get screen state: {e}")
                updates["current_screen"] = "Error getting screen state"
        
        return updates
    
    # ---- Node: Reason ----
    def reason(state: AgentState) -> dict:
        """LLM reasoning step - decide what to do next."""
        logger.debug(f"Reason node executing, step {state.get('step_count', 0)}")
        
        messages = list(state.get("messages", []))
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
        
        # Add context about current screen if available
        screen = state.get("current_screen", "")
        current_app = state.get("current_app", "")
        last_result = state.get("last_result", "")
        last_action = state.get("last_action", "")
        
        if state.get("step_count", 0) > 1:
            # Build context message
            context_parts = []
            
            if last_action and last_result:
                context_parts.append(f"[Previous action]: {last_action}")
                context_parts.append(f"[Result]: {last_result}")
                
                # Check if the task appears complete
                if "Opened" in last_result and "open" in state.get("user_goal", "").lower():
                    context_parts.append("[Status]: ✓ App was successfully opened. Respond to confirm completion.")
            
            if current_app:
                context_parts.append(f"[Current app on screen]: {current_app}")
            
            if screen:
                # Only include screen state if needed (truncate if long)
                screen_summary = screen[:1500] if len(screen) > 1500 else screen
                context_parts.append(f"[Screen UI elements]:\n{screen_summary}")
            
            if context_parts:
                messages.append(HumanMessage(content="\n".join(context_parts)))
        
        try:
            response = llm_with_tools.invoke(messages)
            logger.debug(f"LLM response: {response.content[:200] if response.content else 'No content'}")
            logger.debug(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}")
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {
                "messages": [AIMessage(content=f"I encountered an error: {e}")],
                "status": "error"
            }
    
    # ---- Node: Act ----
    def act(state: AgentState) -> dict:
        """Execute tool calls from the LLM."""
        logger.debug("Act node executing")
        
        messages = state.get("messages", [])
        if not messages:
            return {"status": "error", "last_result": "No messages to process"}
        
        last_message = messages[-1]
        
        # Check if there are tool calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.debug("No tool calls, agent is done reasoning")
            return {"status": "completed"}
        
        # Execute each tool call
        tool_results = []
        tool_map = {t.name: t for t in TOOLS}
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"Executing tool: {tool_name}({tool_args})")
            
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    logger.debug(f"Tool result: {result[:200] if len(result) > 200 else result}")
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"
                    logger.error(result)
            else:
                result = f"Unknown tool: {tool_name}"
            
            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        
        last_result = tool_results[-1].content if tool_results else ""
        
        return {
            "messages": tool_results,
            "last_action": last_message.tool_calls[0]["name"] if last_message.tool_calls else None,
            "last_result": last_result,
            "status": "running"
        }
    
    # ---- Node: Human Review (optional) ----
    def human_review(state: AgentState) -> dict:
        """Pause for human confirmation before certain actions."""
        messages = state.get("messages", [])
        if not messages:
            return {}
        
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Store pending action for review
            pending = json.dumps(last_message.tool_calls[0], indent=2)
            return {"pending_confirmation": pending, "status": "needs_input"}
        
        return {}
    
    # ---- Routing Logic ----
    def should_continue(state: AgentState) -> str:
        """Determine next step based on state."""
        status = state.get("status", "running")
        
        if status in ("completed", "error"):
            return "end"
        
        if status == "needs_input":
            return "wait_for_input"
        
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # If last message is from AI with tool calls, execute them
        if isinstance(last_message, AIMessage):
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "act"
            else:
                # AI responded without tools - done
                return "end"
        
        # If last message is tool result, go back to reasoning
        if isinstance(last_message, ToolMessage):
            # Check step limit
            if state.get("step_count", 0) >= 20:
                return "end"
            return "perceive"
        
        return "end"
    
    def after_act(state: AgentState) -> str:
        """Route after action execution."""
        if state.get("status") == "completed":
            return "end"
        if state.get("step_count", 0) >= 20:
            return "end"
        return "perceive"
    
    # ---- Build the Graph ----
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("perceive", perceive)
    workflow.add_node("reason", reason)
    workflow.add_node("act", act)
    
    if confirm_actions:
        workflow.add_node("human_review", human_review)
    
    # Set entry point
    workflow.set_entry_point("perceive")
    
    # Add edges
    workflow.add_edge("perceive", "reason")
    
    if confirm_actions:
        workflow.add_conditional_edges(
            "reason",
            should_continue,
            {
                "act": "human_review",
                "end": END,
                "perceive": "perceive",
            }
        )
        workflow.add_edge("human_review", "act")
    else:
        workflow.add_conditional_edges(
            "reason", 
            should_continue,
            {
                "act": "act",
                "end": END,
                "perceive": "perceive",
            }
        )
    
    workflow.add_conditional_edges(
        "act",
        after_act,
        {
            "perceive": "perceive",
            "end": END,
        }
    )
    
    return workflow


# ============================================================================
# AGENT CLASS
# ============================================================================

class AndroidAgent:
    """High-level Android agent using LangGraph."""
    
    def __init__(
        self,
        device_ip: Optional[str] = None,
        port: int = 5555,
        model: str = "gpt-4o-mini",
        local_llm_url: Optional[str] = None,
        use_usb: bool = False,
        confirm_actions: bool = False,
        db_path: str = "agent_memory.db",
    ):
        self.device_ip = device_ip
        self.port = port
        self.model = model
        self.local_llm_url = local_llm_url
        self.use_usb = use_usb
        self.confirm_actions = confirm_actions
        self.db_path = db_path
        
        # Components (initialized on connect)
        self.connection: Optional[ConnectionManager] = None
        self.app_library: Optional[AppLibrary] = None
        self.vision: Optional[VisionModule] = None
        self.executor: Optional[ActionExecutor] = None
        self.graph = None
        self.checkpointer = None
    
    def connect(self) -> bool:
        """Connect to the device and initialize all components."""
        global _device_connection, _app_library, _vision, _executor
        
        print("\n" + "="*50)
        print("ANDROID AGENT - Initializing")
        print("="*50 + "\n")
        
        # Connect to device
        self.connection = ConnectionManager(
            device_ip=self.device_ip,
            port=self.port,
            use_usb=self.use_usb
        )
        
        if not self.connection.connect():
            print("❌ Failed to connect to device")
            return False
        
        device = self.connection.get_device()
        device_address = self.connection.device_address
        print(f"✓ Connected: {device.serial}")
        
        # Initialize components
        self.app_library = AppLibrary(device_address)
        self.app_library.fetch_installed_apps()
        
        self.vision = VisionModule(device)
        self.executor = ActionExecutor(device, self.vision, self.app_library)
        
        # Set global references for tools
        _device_connection = self.connection
        _app_library = self.app_library
        _vision = self.vision
        _executor = self.executor
        
        # Build the graph
        workflow = create_agent_graph(
            model_name=self.model,
            base_url=self.local_llm_url,
            confirm_actions=self.confirm_actions,
        )
        
        # Setup persistence
        self.checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        
        print("\n✓ All components initialized")
        return True
    
    def run_task(self, goal: str, thread_id: str = "default") -> str:
        """Run a task and return the result."""
        print("\n" + "-"*50)
        print(f"TASK: {goal}")
        print("-"*50)
        
        # Get app context for the goal
        app_context = self._get_app_context(goal)
        if app_context:
            print(f"\n📱 Relevant apps found")
        
        # Build initial message
        initial_message = goal
        if app_context:
            initial_message = f"{goal}\n\n[Installed apps context]\n{app_context}"
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=initial_message)],
            "user_goal": goal,
            "current_screen": "",
            "current_app": "",
            "app_context": app_context,
            "step_count": 0,
            "last_action": None,
            "last_result": None,
            "status": "running",
            "pending_confirmation": None,
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the graph
        final_state = None
        for step, state in enumerate(self.graph.stream(initial_state, config)):
            # Print progress
            node_name = list(state.keys())[0]
            node_state = state[node_name]
            
            if node_name == "perceive":
                app = node_state.get("current_app", "")
                step_num = node_state.get("step_count", 0)
                print(f"\n--- Step {step_num} ---")
                if app:
                    print(f"📱 Current app: {app}")
            
            elif node_name == "reason":
                messages = node_state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage):
                        if last_msg.content:
                            print(f"🤔 {last_msg.content[:100]}...")
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                print(f"🔧 Planning: {tc['name']}({tc['args']})")
            
            elif node_name == "act":
                action = node_state.get("last_action")
                result = node_state.get("last_result", "")
                if action:
                    print(f"✅ Executed: {action}")
                    if result and len(result) < 100:
                        print(f"   Result: {result}")
            
            final_state = node_state
            
            # Check for completion
            if node_state.get("status") == "completed":
                break
            
            # Safety limit
            if step > 50:
                print("⚠️ Max iterations reached")
                break
        
        # Get final response
        if final_state:
            messages = final_state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 {msg.content}")
                    return msg.content
        
        return "Task completed"
    
    def _get_app_context(self, goal: str) -> str:
        """Find relevant apps for the goal."""
        if not self.app_library:
            return ""
        
        goal_lower = goal.lower()
        
        # Check if it's a question about apps
        is_app_question = any(word in goal_lower for word in ["how many", "which", "what", "list", "apps", "installed"])
        if is_app_question:
            return self.app_library.get_apps_summary(max_apps=200)
        
        # Find specific apps mentioned
        words = goal_lower.split()
        relevant_apps = []
        
        for word in words:
            if len(word) > 2:
                matches = self.app_library.fuzzy_find_app(word, threshold=70)
                for app in matches[:2]:
                    if app not in relevant_apps:
                        relevant_apps.append(app)
        
        if not relevant_apps:
            return ""
        
        lines = ["Matching installed apps:"]
        for app in relevant_apps[:10]:
            lines.append(f"  - {app.common_name}: {app.package_name}")
        return "\n".join(lines)
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("="*50)
        print("Enter your commands. Type 'quit' to exit.")
        print("Special: 'apps' (list apps), 'screen' (show UI)")
        print()
        
        thread_id = "interactive"
        
        while True:
            try:
                goal = input("\n🤖 What should I do? > ").strip()
                
                if not goal:
                    continue
                
                if goal.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                
                if goal.lower() == "apps":
                    print("\nInstalled Apps:")
                    print(self.app_library.get_apps_summary())
                    continue
                
                if goal.lower() == "screen":
                    print("\nCurrent Screen:")
                    print(self.vision.get_ui_summary())
                    continue
                
                self.run_task(goal, thread_id)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                logger.exception(f"Error: {e}")
                print(f"\n❌ Error: {e}")
    
    def disconnect(self):
        """Disconnect from the device."""
        if self.connection:
            self.connection.disconnect()
