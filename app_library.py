"""App Library for managing installed Android applications."""

import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional
from thefuzz import fuzz, process


@dataclass
class AppInfo:
    """Information about an installed app."""
    package_name: str
    common_name: str
    
    def __str__(self):
        return f"{self.common_name} ({self.package_name})"


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
    
    def __init__(self, device_address: str):
        self.device_address = device_address
        self.apps: List[AppInfo] = []
    
    def _run_adb(self, args: list[str]) -> tuple[bool, str]:
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
        """Fetch all user-installed apps from the device."""
        print("Fetching installed apps...")
        
        # Get third-party packages (-3 flag)
        success, output = self._run_adb(["shell", "cmd", "package", "list", "packages", "-3"])
        
        if not success:
            print(f"Failed to fetch packages: {output}")
            return []
        
        self.apps = []
        
        for line in output.splitlines():
            if line.startswith("package:"):
                package_name = line[8:].strip()
                
                # Skip ignored packages
                if self._should_ignore(package_name):
                    continue
                
                common_name = self._package_to_common_name(package_name)
                self.apps.append(AppInfo(package_name, common_name))
        
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
                    if package_name in useful_system_apps:
                        common_name = self._package_to_common_name(package_name)
                        # Avoid duplicates
                        if not any(app.package_name == package_name for app in self.apps):
                            self.apps.append(AppInfo(package_name, common_name))
        
        print(f"✓ Found {len(self.apps)} apps")
        return self.apps
    
    def fuzzy_find_app(self, query: str, threshold: int = 60) -> List[AppInfo]:
        """Find apps matching a fuzzy query."""
        if not self.apps:
            return []
        
        results = []
        query_lower = query.lower()
        
        for app in self.apps:
            # Check common name match
            common_score = fuzz.partial_ratio(query_lower, app.common_name.lower())
            # Check package name match
            package_score = fuzz.partial_ratio(query_lower, app.package_name.lower())
            
            best_score = max(common_score, package_score)
            
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
            lines.append(f"- {app.common_name}: {app.package_name}")
        
        if len(self.apps) > max_apps:
            lines.append(f"... and {len(self.apps) - max_apps} more apps")
        
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
