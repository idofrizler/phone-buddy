"""Vision module for parsing Android UI hierarchy."""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import base64
import io


@dataclass
class UIElement:
    """Represents a UI element on screen."""
    uid: int
    element_type: str
    text: str
    content_desc: str
    resource_id: str
    bounds: Tuple[int, int, int, int]  # left, top, right, bottom
    clickable: bool
    scrollable: bool
    enabled: bool
    focused: bool
    
    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "type": self.element_type,
            "text": self.text,
            "content_desc": self.content_desc,
            "resource_id": self.resource_id.split("/")[-1] if "/" in self.resource_id else self.resource_id,
            "bounds": list(self.bounds),
            "clickable": self.clickable,
            "scrollable": self.scrollable,
        }
    
    def get_center(self) -> Tuple[int, int]:
        """Get center coordinates of the element."""
        left, top, right, bottom = self.bounds
        return ((left + right) // 2, (top + bottom) // 2)
    
    def get_description(self) -> str:
        """Get a human-readable description of this element."""
        parts = []
        if self.text:
            parts.append(f'"{self.text}"')
        if self.content_desc:
            parts.append(f'[{self.content_desc}]')
        if self.resource_id:
            short_id = self.resource_id.split("/")[-1]
            parts.append(f'({short_id})')
        
        desc = " ".join(parts) if parts else self.element_type
        
        attrs = []
        if self.clickable:
            attrs.append("clickable")
        if self.scrollable:
            attrs.append("scrollable")
        
        if attrs:
            desc += f" [{', '.join(attrs)}]"
        
        return desc


class VisionModule:
    """Handles UI perception through XML parsing and screenshots."""
    
    def __init__(self, device):
        self.device = device
        self.current_elements: List[UIElement] = []
        self.uid_counter = 0
    
    def _parse_bounds(self, bounds_str: str) -> Tuple[int, int, int, int]:
        """Parse bounds string like '[0,0][1080,1920]' to tuple."""
        match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
        if match:
            return tuple(map(int, match.groups()))
        return (0, 0, 0, 0)
    
    def _is_meaningful_element(self, node: ET.Element) -> bool:
        """Check if an element is meaningful for interaction."""
        text = node.get("text", "")
        content_desc = node.get("content-desc", "")
        resource_id = node.get("resource-id", "")
        clickable = node.get("clickable", "false") == "true"
        scrollable = node.get("scrollable", "false") == "true"
        enabled = node.get("enabled", "true") == "true"
        
        # Element must be enabled
        if not enabled:
            return False
        
        # Has text or description
        has_content = bool(text or content_desc or resource_id)
        
        # Is interactive
        is_interactive = clickable or scrollable
        
        # Element types that are usually meaningful
        meaningful_types = [
            "android.widget.Button",
            "android.widget.EditText",
            "android.widget.TextView",
            "android.widget.ImageButton",
            "android.widget.ImageView",
            "android.widget.CheckBox",
            "android.widget.Switch",
            "android.widget.RadioButton",
            "android.widget.Spinner",
            "android.widget.SeekBar",
            "android.view.View",
            "android.widget.LinearLayout",
            "android.widget.FrameLayout",
            "android.widget.RelativeLayout",
            "androidx.recyclerview.widget.RecyclerView",
            "android.widget.ScrollView",
            "android.widget.ListView",
        ]
        
        class_name = node.get("class", "")
        is_meaningful_type = any(t in class_name for t in meaningful_types)
        
        # Include if it has content or is interactive
        return has_content or (is_interactive and is_meaningful_type)
    
    def _parse_node(self, node: ET.Element) -> Optional[UIElement]:
        """Parse an XML node into a UIElement."""
        if not self._is_meaningful_element(node):
            return None
        
        self.uid_counter += 1
        
        class_name = node.get("class", "unknown")
        # Simplify class name
        simple_type = class_name.split(".")[-1]
        
        bounds = self._parse_bounds(node.get("bounds", "[0,0][0,0]"))
        
        # Skip elements with zero size
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return None
        
        return UIElement(
            uid=self.uid_counter,
            element_type=simple_type,
            text=node.get("text", ""),
            content_desc=node.get("content-desc", ""),
            resource_id=node.get("resource-id", ""),
            bounds=bounds,
            clickable=node.get("clickable", "false") == "true",
            scrollable=node.get("scrollable", "false") == "true",
            enabled=node.get("enabled", "true") == "true",
            focused=node.get("focused", "false") == "true",
        )
    
    def _traverse_tree(self, node: ET.Element, elements: List[UIElement]):
        """Recursively traverse the UI tree and collect elements."""
        element = self._parse_node(node)
        if element:
            elements.append(element)
        
        for child in node:
            self._traverse_tree(child, elements)
    
    def get_ui_state(self) -> dict:
        """Get the current UI state as simplified JSON."""
        self.uid_counter = 0
        self.current_elements = []
        
        try:
            # Get XML hierarchy
            xml_content = self.device.dump_hierarchy()
            root = ET.fromstring(xml_content)
            
            self._traverse_tree(root, self.current_elements)
            
            # Get current package using ADB (more reliable than uiautomator2)
            import subprocess
            try:
                result = subprocess.run(
                    ["adb", "shell", "dumpsys window | grep -E 'mCurrentFocus|mFocusedApp' | head -2"],
                    capture_output=True, text=True, timeout=5, shell=False
                )
                # Actually run properly
                result = subprocess.run(
                    ["adb", "shell", "dumpsys", "window", "windows"],
                    capture_output=True, text=True, timeout=5
                )
                output = result.stdout
                # Extract package from mCurrentFocus or mFocusedApp
                import re
                match = re.search(r'mCurrentFocus.*?([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/', output)
                if not match:
                    match = re.search(r'mFocusedApp.*?([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/', output)
                current_package = match.group(1) if match else "unknown"
            except Exception:
                # Fallback to uiautomator2
                current_app = self.device.app_current()
                current_package = current_app.get("package", "unknown")
            
            # Build simplified state
            state = {
                "package": current_package,
                "element_count": len(self.current_elements),
                "elements": [el.to_dict() for el in self.current_elements if el.clickable or el.scrollable or el.text or el.content_desc]
            }
            
            return state
            
        except Exception as e:
            return {
                "error": str(e),
                "package": "unknown",
                "activity": "unknown",
                "elements": []
            }
    
    def get_ui_summary(self) -> str:
        """Get a text summary of the current UI for the LLM."""
        state = self.get_ui_state()
        
        if "error" in state:
            return f"Error getting UI: {state['error']}"
        
        lines = [
            f"Current App: {state['package']}",
            f"",
            "Interactive Elements:",
        ]
        
        for el in state["elements"]:
            uid = el["uid"]
            el_type = el["type"]
            text = el.get("text", "")
            desc = el.get("content_desc", "")
            res_id = el.get("resource_id", "")
            
            # Build description
            parts = [f"[{uid}]"]
            
            if text:
                parts.append(f'"{text}"')
            elif desc:
                parts.append(f'[{desc}]')
            elif res_id:
                parts.append(f'({res_id})')
            else:
                parts.append(el_type)
            
            if el.get("clickable"):
                parts.append("• clickable")
            if el.get("scrollable"):
                parts.append("• scrollable")
            
            lines.append("  " + " ".join(parts))
        
        return "\n".join(lines)
    
    def get_element_by_uid(self, uid: int) -> Optional[UIElement]:
        """Get an element by its UID."""
        for el in self.current_elements:
            if el.uid == uid:
                return el
        return None
    
    def take_screenshot(self) -> bytes:
        """Take a screenshot and return as bytes."""
        try:
            img = self.device.screenshot()
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return b""
    
    def screenshot_base64(self) -> str:
        """Take a screenshot and return as base64 string."""
        img_bytes = self.take_screenshot()
        if img_bytes:
            return base64.b64encode(img_bytes).decode("utf-8")
        return ""
