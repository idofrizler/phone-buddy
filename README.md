# Phone Buddy - Headless Android Agent

A Python-based agent that controls your Android phone wirelessly using natural language commands. It uses ADB over WiFi, UI automation, and LLM reasoning to accomplish complex tasks.

## Features

- **Wireless Control**: Connect to your Android device over WiFi using ADB
- **Natural Language**: Describe tasks in plain English (e.g., "Open Spotify and play my liked songs")
- **Smart App Launch**: Directly launches any installed app without needing to find icons
- **UI Understanding**: Parses the Android UI hierarchy to understand what's on screen
- **LLM-Powered**: Uses GPT-4o-mini (or local LLMs) for intelligent decision-making

## Prerequisites

1. **Android SDK Platform Tools** - ADB must be installed and in your PATH
   ```bash
   # macOS
   brew install android-platform-tools
   
   # Ubuntu/Debian
   sudo apt install adb
   
   # Or download from: https://developer.android.com/studio/releases/platform-tools
   ```

2. **Python 3.10+**

3. **OpenAI API Key** (or a local LLM server)
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

## Installation

```bash
cd phone-buddy
pip install -r requirements.txt
```

## Device Setup

### One-Time Setup (USB Required)

1. Enable **Developer Options** on your Android phone:
   - Go to Settings > About Phone
   - Tap "Build Number" 7 times

2. Enable **USB Debugging**:
   - Go to Settings > Developer Options
   - Enable "USB Debugging"

3. Enable **Wireless Debugging** (Android 11+):
   - Go to Settings > Developer Options
   - Enable "Wireless Debugging"
   - Note your phone's IP address

4. Connect via USB and enable TCP/IP mode:
   ```bash
   python main.py <phone-ip> --setup-tcpip
   ```

5. Disconnect USB - you can now use wireless ADB!

## Usage

### Interactive Mode

```bash
python main.py 192.168.1.100
```

This starts an interactive session where you can type commands:

```
🤖 What should I do? > Open WhatsApp and send "Hello" to Mom

--- Step 1 ---
Screen: Current App: com.android.launcher3
Action: open_app (com.whatsapp)
Reasoning: Opening WhatsApp directly to send a message
Result: Opened com.whatsapp

--- Step 2 ---
Screen: Current App: com.whatsapp
Action: click (uid=15)
Reasoning: Clicking search to find contact
...
```

### Single Task Mode

```bash
python main.py 192.168.1.100 --task "Open Chrome and search for weather"
```

### Using Local LLM

```bash
# With LM Studio, Ollama, or similar
python main.py 192.168.1.100 --local-llm http://localhost:1234/v1
```

### Special Commands

In interactive mode:
- `apps` - List all installed apps
- `screen` - Show current UI elements
- `quit` - Exit

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│                   AndroidAgent                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Connection   │  │  App Library │  │   Vision     │  │
│  │   Manager    │  │              │  │   Module     │  │
│  │              │  │ - Fetch apps │  │              │  │
│  │ - ADB TCP/IP │  │ - Fuzzy find │  │ - XML parse  │  │
│  │ - Connect    │  │ - Launch     │  │ - Screenshot │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │    Brain     │  │   Executor   │                    │
│  │              │  │              │                    │
│  │ - LLM reason │  │ - Click      │                    │
│  │ - Actions    │  │ - Type       │                    │
│  │              │  │ - Scroll     │                    │
│  └──────────────┘  └──────────────┘                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Supported Actions

| Action | Description |
|--------|-------------|
| `click` | Tap on a UI element by UID |
| `type` | Enter text into focused field |
| `scroll` | Scroll up/down/left/right |
| `open_app` | Launch any installed app directly |
| `back` | Press the back button |
| `home` | Press the home button |
| `wait` | Wait for UI to settle |
| `done` | Mark task as complete |

## Examples

```bash
# Open an app
python main.py 192.168.1.100 -t "Open YouTube"

# Send a message
python main.py 192.168.1.100 -t "Open WhatsApp and send 'I'm on my way' to John"

# Search something
python main.py 192.168.1.100 -t "Search for nearby coffee shops on Google Maps"

# Change settings
python main.py 192.168.1.100 -t "Turn on airplane mode"
```

## Troubleshooting

### Connection Issues

```bash
# Check if device is reachable
adb devices

# If not listed, try:
adb kill-server
adb start-server
adb connect 192.168.1.100:5555
```

### UIAutomator2 Issues

The first connection may install UIAutomator2 on your device. Allow any installation prompts.

### LLM Errors

Make sure `OPENAI_API_KEY` is set, or use `--local-llm` for local inference.

## License

MIT
