"""App Library for managing installed Android applications."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from thefuzz import fuzz, process


@dataclass
class AppInfo:
    """Information about an installed app."""
    package_name: str
    common_name: str  # Derived from package name
    display_name: Optional[str] = None  # Actual app label from APK
    
    def __str__(self):
        name = self.display_name or self.common_name
        return f"{name} ({self.package_name})"
    
    @property
    def best_name(self) -> str:
        """Get the best available name for this app."""
        return self.display_name or self.common_name


class AppLibrary:
    """Fetches and caches installed apps with fuzzy search capabilities."""
    
    # Package prefixes to ignore (system/bloatware)
    IGNORED_PREFIXES = [
        "com.android.",
        "com.google.android.inputmethod",
        "com.google.android.gms",
        "com.google.android.gsf",
        "com.google.android.providers",
        "com.google.android.ext.",
        "com.google.android.onetimeinitializer",
        "com.google.android.configupdater",
        "com.google.android.partnersetup",
        "com.google.android.printservice",
        "com.google.android.syncadapters",
        "com.google.android.feedback",
        "com.google.android.backuptransport",
        "com.samsung.",
        "com.sec.",
        "com.qualcomm.",
        "com.mediatek.",
        "org.codeaurora.",
    ]
    
    # Known package name to common name mappings
    KNOWN_APPS = {
        "com.whatsapp": "WhatsApp",
        "com.instagram.android": "Instagram",
        "com.facebook.katana": "Facebook",
        "com.facebook.orca": "Messenger",
        "com.twitter.android": "Twitter",
        "com.spotify.music": "Spotify",
        "com.netflix.mediaclient": "Netflix",
        "com.google.android.youtube": "YouTube",
        "com.snapchat.android": "Snapchat",
        "com.zhiliaoapp.musically": "TikTok",
        "com.reddit.frontpage": "Reddit",
        "com.linkedin.android": "LinkedIn",
        "com.pinterest": "Pinterest",
        "com.discord": "Discord",
        "com.slack": "Slack",
        "org.telegram.messenger": "Telegram",
        "com.viber.voip": "Viber",
        "com.skype.raider": "Skype",
        "com.amazon.mShop.android.shopping": "Amazon",
        "com.ebay.mobile": "eBay",
        "com.ubercab": "Uber",
        "com.lyft.android": "Lyft",
        "com.airbnb.android": "Airbnb",
        "com.booking": "Booking.com",
        "com.google.android.apps.maps": "Google Maps",
        "com.waze": "Waze",
        "com.google.android.apps.photos": "Google Photos",
        "com.google.android.gm": "Gmail",
        "com.google.android.apps.docs": "Google Drive",
        "com.microsoft.office.outlook": "Outlook",
        "com.microsoft.teams": "Microsoft Teams",
        "com.dropbox.android": "Dropbox",
        "com.evernote": "Evernote",
        "com.todoist": "Todoist",
        "com.notion.id": "Notion",
        "com.duolingo": "Duolingo",
        "com.calm.android": "Calm",
        "com.headspace.android": "Headspace",
        "com.strava": "Strava",
        "com.nike.plusgps": "Nike Run Club",
        "com.fitbit.FitbitMobile": "Fitbit",
        "com.paypal.android.p2pmobile": "PayPal",
        "com.venmo": "Venmo",
        "com.robinhood.android": "Robinhood",
        "com.coinbase.android": "Coinbase",
    }
    
    # Cache file for app labels
    CACHE_DIR = Path.home() / ".cache" / "phone-buddy"
    CACHE_FILE = CACHE_DIR / "app_labels.json"
    
    def __init__(self, device_address: str):
        self.device_address = device_address
        self.apps: List[AppInfo] = []
        self.label_cache: Dict[str, str] = {}
        self._aapt_path = self._find_aapt()
        self._load_label_cache()
    
    def _find_aapt(self) -> Optional[str]:
        """Find aapt binary on the system."""
        # Check if aapt is in PATH
        aapt = shutil.which("aapt")
        if aapt:
            return aapt
        
        # Check common Android SDK locations
        possible_paths = [
            "/opt/homebrew/share/android-commandlinetools/build-tools/34.0.0/aapt",
            "/opt/homebrew/share/android-commandlinetools/build-tools/33.0.0/aapt",
            os.path.expanduser("~/Library/Android/sdk/build-tools/34.0.0/aapt"),
            os.path.expanduser("~/Library/Android/sdk/build-tools/33.0.0/aapt"),
        ]
        
        # Also check ANDROID_HOME
        android_home = os.environ.get("ANDROID_HOME", "")
        if android_home:
            build_tools = Path(android_home) / "build-tools"
            if build_tools.exists():
                for version_dir in sorted(build_tools.iterdir(), reverse=True):
                    aapt_path = version_dir / "aapt"
                    if aapt_path.exists():
                        return str(aapt_path)
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_label_cache(self):
        """Load cached app labels from disk."""
        if self.CACHE_FILE.exists():
            try:
                with open(self.CACHE_FILE, "r") as f:
                    self.label_cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.label_cache = {}
    
    def _save_label_cache(self):
        """Save app labels cache to disk."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.CACHE_FILE, "w") as f:
                json.dump(self.label_cache, f, indent=2)
        except IOError:
            pass
    
    def _run_adb(self, args: List[str]) -> tuple[bool, str]:
        """Run an ADB command targeting the specific device."""
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_address] + args,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout.strip()
        except Exception as e:
            return False, str(e)
    
    def _get_app_label(self, package_name: str) -> Optional[str]:
        """Get the display label for an app using aapt."""
        # Check cache first
        if package_name in self.label_cache:
            return self.label_cache[package_name]
        
        if not self._aapt_path:
            return None
        
        try:
            # Get APK path
            success, output = self._run_adb(["shell", "pm", "path", package_name])
            if not success or not output:
                return None
            
            # Get first APK path (base.apk)
            apk_path = output.split("\n")[0].replace("package:", "").strip()
            
            # Pull APK to temp location
            with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                result = subprocess.run(
                    ["adb", "-s", self.device_address, "pull", apk_path, tmp_path],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode != 0:
                    return None
                
                # Use aapt to get label
                result = subprocess.run(
                    [self._aapt_path, "dump", "badging", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.startswith("application-label:"):
                            label = line.split(":", 1)[1].strip().strip("'\"")
                            self.label_cache[package_name] = label
                            return label
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                    
        except Exception:
            pass
        
        return None
    
    def _package_to_common_name(self, package_name: str) -> str:
        """Convert a package name to a human-readable common name."""
        # Check known mappings first
        if package_name in self.KNOWN_APPS:
            return self.KNOWN_APPS[package_name]
        
        # Extract the meaningful part of the package name
        parts = package_name.split(".")
        
        # Usually the app name is in the last 1-2 segments
        if len(parts) >= 2:
            # Skip common prefixes like com, org, net
            meaningful_parts = [p for p in parts[1:] if p not in 
                              ["android", "app", "apps", "mobile", "client"]]
            if meaningful_parts:
                # Take the most meaningful part (usually the last non-generic one)
                name = meaningful_parts[-1] if len(meaningful_parts) == 1 else meaningful_parts[0]
                # Convert camelCase or snake_case to Title Case
                name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
                name = name.replace("_", " ").replace("-", " ")
                return name.title()
        
        # Fallback: just use the last part
        return parts[-1].title() if parts else package_name
    
    def _should_ignore(self, package_name: str) -> bool:
        """Check if a package should be ignored."""
        return any(package_name.startswith(prefix) for prefix in self.IGNORED_PREFIXES)
    
    def fetch_installed_apps(self) -> List[AppInfo]:
        """Fetch all user-installed apps from the device.
        
        Labels are fetched via aapt and cached to disk. First run will be slow,
        subsequent runs will be fast as labels are loaded from cache.
        """
        print("Fetching installed apps...")
        
        # Get third-party packages (-3 flag)
        success, output = self._run_adb(["shell", "cmd", "package", "list", "packages", "-3"])
        
        if not success:
            print(f"Failed to fetch packages: {output}")
            return []
        
        self.apps = []
        packages_to_process = []
        
        for line in output.splitlines():
            if line.startswith("package:"):
                package_name = line[8:].strip()
                
                # Skip ignored packages
                if self._should_ignore(package_name):
                    continue
                
                packages_to_process.append(package_name)
        
        # Also fetch some useful system apps (Google apps, etc.)
        success, output = self._run_adb(["shell", "cmd", "package", "list", "packages", "-s"])
        if success:
            useful_system_apps = [
                "com.google.android.youtube",
                "com.google.android.apps.maps",
                "com.google.android.apps.photos",
                "com.google.android.gm",
                "com.google.android.apps.docs",
                "com.google.android.calendar",
                "com.google.android.contacts",
                "com.google.android.dialer",
                "com.google.android.apps.messaging",
            ]
            for line in output.splitlines():
                if line.startswith("package:"):
                    package_name = line[8:].strip()
                    if package_name in useful_system_apps and package_name not in packages_to_process:
                        packages_to_process.append(package_name)
        
        # Determine which packages need label fetching
        packages_needing_labels = [p for p in packages_to_process if p not in self.label_cache]
        
        # Fetch labels for new packages only
        if packages_needing_labels and self._aapt_path:
            total_new = len(packages_needing_labels)
            print(f"  Fetching labels for {total_new} new apps (cached: {len(packages_to_process) - total_new})...")
            
            for i, package_name in enumerate(packages_needing_labels):
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"    Progress: {i + 1}/{total_new}")
                self._get_app_label(package_name)
            
            # Save updated cache
            self._save_label_cache()
            print(f"  ✓ Cached {total_new} new app labels")
        
        # Build app list with labels from cache
        for package_name in packages_to_process:
            common_name = self._package_to_common_name(package_name)
            display_name = self.label_cache.get(package_name)
            self.apps.append(AppInfo(package_name, common_name, display_name))
        
        print(f"✓ Found {len(self.apps)} apps")
        return self.apps
    
    def fuzzy_find_app(self, query: str, threshold: int = 60) -> List[AppInfo]:
        """Find apps matching a fuzzy query."""
        if not self.apps:
            return []
        
        results = []
        query_lower = query.lower()
        
        for app in self.apps:
            # Check display name match (if available)
            display_score = 0
            if app.display_name:
                display_score = fuzz.partial_ratio(query_lower, app.display_name.lower())
            
            # Check common name match
            common_score = fuzz.partial_ratio(query_lower, app.common_name.lower())
            # Check package name match
            package_score = fuzz.partial_ratio(query_lower, app.package_name.lower())
            
            best_score = max(display_score, common_score, package_score)
            
            if best_score >= threshold:
                results.append((app, best_score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [app for app, score in results]
    
    def get_app_by_package(self, package_name: str) -> Optional[AppInfo]:
        """Get an app by its exact package name."""
        for app in self.apps:
            if app.package_name == package_name:
                return app
        return None
    
    def get_apps_summary(self, max_apps: int = 50) -> str:
        """Get a summary of installed apps for LLM context."""
        if not self.apps:
            return "No apps cached. Call fetch_installed_apps() first."
        
        lines = []
        for app in self.apps[:max_apps]:
            name = app.display_name or app.common_name
            lines.append(f"- {name}: {app.package_name}")
        
        if len(self.apps) > max_apps:
            lines.append(f"... and {len(self.apps) - max_apps} more apps")
        
        return "\n".join(lines)
        
        return "\n".join(lines)
    
    def launch_app(self, device, package_name: str) -> bool:
        """Launch an app using uiautomator2."""
        try:
            device.app_start(package_name)
            print(f"✓ Launched {package_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to launch {package_name}: {e}")
            return False
