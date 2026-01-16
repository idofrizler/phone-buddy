"""Executor module for performing actions on the Android device."""

import time
from typing import Optional
from vision import VisionModule, UIElement
from brain import AgentAction
from app_library import AppLibrary


class ActionExecutor:
    """Executes actions on the Android device."""
    
    def __init__(self, device, vision: VisionModule, app_library: AppLibrary):
        self.device = device
        self.vision = vision
        self.app_library = app_library
    
    def execute(self, action: AgentAction) -> tuple[bool, str]:
        """Execute an action and return success status and message."""
        
        if action.action == "click":
            return self._execute_click(action.target_uid)
        
        elif action.action == "type":
            return self._execute_type(action.text)
        
        elif action.action == "scroll":
            return self._execute_scroll(action.direction)
        
        elif action.action == "open_app":
            return self._execute_open_app(action.app_package)
        
        elif action.action == "back":
            return self._execute_back()
        
        elif action.action == "home":
            return self._execute_home()
        
        elif action.action == "wait":
            time.sleep(2)
            return True, "Waited 2 seconds"
        
        elif action.action == "done":
            return True, "Task completed"
        
        elif action.action == "respond":
            return True, action.message or "No message"
        
        elif action.action == "ask":
            return True, action.message or "No question"
        
        else:
            return False, f"Unknown action: {action.action}"
    
    def _execute_click(self, target_uid: Optional[int]) -> tuple[bool, str]:
        """Click on an element by UID."""
        if target_uid is None:
            return False, "No target_uid provided for click action"
        
        element = self.vision.get_element_by_uid(target_uid)
        if element is None:
            return False, f"Element with UID {target_uid} not found"
        
        x, y = element.get_center()
        
        try:
            self.device.click(x, y)
            desc = element.get_description()
            return True, f"Clicked on {desc} at ({x}, {y})"
        except Exception as e:
            return False, f"Click failed: {e}"
    
    def _execute_type(self, text: Optional[str]) -> tuple[bool, str]:
        """Type text into the currently focused element."""
        if not text:
            return False, "No text provided for type action"
        
        try:
            # Clear existing text and type new text
            self.device.clear_text()
            self.device.send_keys(text)
            return True, f"Typed: {text}"
        except Exception as e:
            return False, f"Type failed: {e}"
    
    def _execute_scroll(self, direction: Optional[str]) -> tuple[bool, str]:
        """Scroll in the specified direction."""
        direction = direction or "down"
        
        try:
            # Get screen dimensions
            info = self.device.info
            width = info["displayWidth"]
            height = info["displayHeight"]
            
            # Calculate scroll coordinates
            center_x = width // 2
            center_y = height // 2
            
            scroll_distance = height // 3
            
            if direction == "down":
                self.device.swipe(center_x, center_y + scroll_distance, center_x, center_y - scroll_distance, duration=0.3)
            elif direction == "up":
                self.device.swipe(center_x, center_y - scroll_distance, center_x, center_y + scroll_distance, duration=0.3)
            elif direction == "left":
                self.device.swipe(center_x + scroll_distance, center_y, center_x - scroll_distance, center_y, duration=0.3)
            elif direction == "right":
                self.device.swipe(center_x - scroll_distance, center_y, center_x + scroll_distance, center_y, duration=0.3)
            else:
                return False, f"Invalid scroll direction: {direction}"
            
            return True, f"Scrolled {direction}"
        except Exception as e:
            return False, f"Scroll failed: {e}"
    
    def _execute_open_app(self, package_name: Optional[str]) -> tuple[bool, str]:
        """Open an app by package name."""
        if not package_name:
            return False, "No app_package provided for open_app action"
        
        try:
            success = self.app_library.launch_app(self.device, package_name)
            if success:
                time.sleep(2)  # Wait for app to launch
                return True, f"Opened {package_name}"
            else:
                return False, f"Failed to open {package_name}"
        except Exception as e:
            return False, f"Open app failed: {e}"
    
    def _execute_back(self) -> tuple[bool, str]:
        """Press the back button."""
        try:
            self.device.press("back")
            return True, "Pressed back"
        except Exception as e:
            return False, f"Back failed: {e}"
    
    def _execute_home(self) -> tuple[bool, str]:
        """Press the home button."""
        try:
            self.device.press("home")
            return True, "Pressed home"
        except Exception as e:
            return False, f"Home failed: {e}"
