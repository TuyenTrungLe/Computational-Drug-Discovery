"""
Base Page class for Streamlit pages following OOP principles
"""

from abc import ABC, abstractmethod
import streamlit as st
from typing import Optional, Dict, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class BasePage(ABC):
    """
    Abstract base class for all Streamlit pages
    Provides common functionality and enforces consistent structure
    """
    
    def __init__(self, title: str, icon: str = "📄"):
        """
        Initialize page
        
        Args:
            title: Page title
            icon: Page icon emoji
        """
        self.title = title
        self.icon = icon
        self.page_config = {}
        
    @abstractmethod
    def render_content(self):
        """
        Render main page content - must be implemented by subclasses
        """
        pass
    
    def render_header(self):
        """Render page header"""
        st.title(f"{self.icon} {self.title}")
        self._add_divider()
    
    def render_footer(self):
        """Render page footer (optional)"""
        pass
    
    def render(self):
        """Main render method - template pattern"""
        self.render_header()
        self.render_content()
        self.render_footer()
    
    def _add_divider(self):
        """Add visual divider"""
        st.markdown("---")
    
    def show_error(self, message: str):
        """Display error message"""
        st.error(f"❌ {message}")
    
    def show_success(self, message: str):
        """Display success message"""
        st.success(f"✓ {message}")
    
    def show_warning(self, message: str):
        """Display warning message"""
        st.warning(f"⚠️ {message}")
    
    def show_info(self, message: str):
        """Display info message"""
        st.info(f"ℹ️ {message}")
    
    def create_download_button(self, label: str, data: Any, filename: str, 
                               mime: str = "text/csv", **kwargs):
        """
        Create consistent download button
        
        Args:
            label: Button label
            data: Data to download
            filename: Download filename
            mime: MIME type
            **kwargs: Additional button parameters
        """
        return st.download_button(
            label=f"📥 {label}",
            data=data,
            file_name=filename,
            mime=mime,
            use_container_width=kwargs.get('use_container_width', True),
            **{k: v for k, v in kwargs.items() if k != 'use_container_width'}
        )
    
    def create_metric_card(self, label: str, value: str, delta: Optional[str] = None,
                          help_text: Optional[str] = None):
        """
        Create consistent metric display
        
        Args:
            label: Metric label
            value: Metric value
            delta: Change indicator
            help_text: Help tooltip
        """
        st.metric(
            label=label,
            value=value,
            delta=delta,
            help=help_text
        )
