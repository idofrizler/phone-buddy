#!/usr/bin/env python3
"""
Headless Android Agent - Control your Android phone with natural language.

Uses LangGraph for a mature ReACT loop with persistence and human-in-the-loop support.

Usage:
    python main.py --usb --local-llm http://localhost:11434/v1 --model gpt-oss:20b
    python main.py <device_ip> [--port 5555] [--model gpt-4o-mini]
"""

import argparse
import sys

from graph_agent import AndroidAgent
from connection_manager import ConnectionManager


def main():
    parser = argparse.ArgumentParser(
        description="Headless Android Agent - Control your phone with natural language"
    )
    parser.add_argument(
        "device_ip",
        nargs="?",
        default=None,
        help="IP address of the Android device (not needed with --usb)"
    )
    parser.add_argument(
        "--usb",
        action="store_true",
        help="Connect via USB instead of wireless"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5555,
        help="ADB port (default: 5555)"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--local-llm",
        metavar="URL",
        help="URL for local LLM server (e.g., http://localhost:1234/v1)"
    )
    parser.add_argument(
        "--task", "-t",
        help="Run a single task and exit"
    )
    parser.add_argument(
        "--setup-tcpip",
        action="store_true",
        help="Setup ADB TCP/IP mode (requires USB connection first)"
    )
    
    args = parser.parse_args()
    
    # Validate args
    if not args.usb and not args.device_ip and not args.setup_tcpip:
        parser.error("Either device_ip or --usb is required")
    
    # Handle TCP/IP setup mode
    if args.setup_tcpip:
        conn = ConnectionManager(args.device_ip, args.port)
        conn.setup_tcpip()
        print(f"\nNow run: python main.py {args.device_ip}")
        return
    
    # Create and run agent
    agent = AndroidAgent(
        device_ip=args.device_ip,
        port=args.port,
        model=args.model,
        local_llm_url=args.local_llm,
        use_usb=args.usb,
    )
    
    try:
        if not agent.connect():
            sys.exit(1)
        
        if args.task:
            # Single task mode
            success = agent.run_task(args.task)
            sys.exit(0 if success else 1)
        else:
            # Interactive mode
            agent.interactive_mode()
            
    finally:
        agent.disconnect()


if __name__ == "__main__":
    main()
