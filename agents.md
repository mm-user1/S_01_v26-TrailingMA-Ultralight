## Base rules for project:
Act as an experienced Python and Pinescript developer with trading and crypto algorithmic trading expertise.
IMPORTANT: Work strictly according to the given specifications. Any deviations are prohibited without my explicit consent.
IMPORTANT: The script must be maximally efficient and fast.
IMPORTANT: The GUI must use a light theme.
## UI Design Guidelines - Trading Backtester (DearPyGUI)
### Core Design Principles
#### Color Scheme - Strict Monochrome
All colors specified in DearPyGUI format: (R, G, B, A) where values are 0-255.
- Background (body): (232, 232, 232, 255) - light gray
- Window background: (245, 245, 245, 255) - off-white
- Title bar: (74, 74, 74, 255) - dark gray
- Borders:
	- Primary: (153, 153, 153, 255)
	- Secondary: (187, 187, 187, 255)
	- Tertiary: (204, 204, 204, 255)
- Text colors:
	- Primary: (42, 42, 42, 255) - near black
	- Secondary: (58, 58, 58, 255)
	- Tertiary: (90, 90, 90, 255)
	- Disabled/placeholder: (119, 119, 119, 255)
- Button primary: (74, 74, 74, 255) background, (255, 255, 255, 255) text
- Button secondary: (204, 204, 204, 255) background, (42, 42, 42, 255) text
- Input fields: (255, 255, 255, 255) background with (153, 153, 153, 255) border
- Section backgrounds: (232, 232, 232, 255)
- Hover states: (221, 221, 221, 255)
- Focus border: (90, 90, 90, 255)
- NO COLORS - strictly grayscale only
#### Typography
- Font: Default DearPyGUI font or load custom font (Segoe UI preferred)
- Font sizes:
	- Title bar: 15px
	- Labels: 14px
	- Section titles: 12px (uppercase in text, not font)
	- Input text: 14px
	- Small labels: 13px
### DearPyGUI Window Structure
#### Main Window Configuration
```import dearpygui.dearpygui as dpg

# Window setup
dpg.create_context()
dpg.create_viewport(title="S_01 TrailingMA Backtester", width=840, height=900)
dpg.setup_dearpygui()

# Main window
with dpg.window(label="S_01 TrailingMA Backtester", tag="main_window", 
                width=800, height=850, pos=[20, 20],
                no_resize=True, no_move=False, no_close=False):
    # Content goes here
    pass
```
#### Theme Application
```# Create monochrome theme
with dpg.theme() as monochrome_theme:
    with dpg.theme_component(dpg.mvAll):
        # Window background
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (245, 245, 245, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (245, 245, 245, 255))
        
        # Title bar
        dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (74, 74, 74, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (74, 74, 74, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, (74, 74, 74, 255))
        
        # Borders
        dpg.add_theme_color(dpg.mvThemeCol_Border, (153, 153, 153, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
        dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
        
        # Text
        dpg.add_theme_color(dpg.mvThemeCol_Text, (42, 42, 42, 255))
        
        # Buttons
        dpg.add_theme_color(dpg.mvThemeCol_Button, (74, 74, 74, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (58, 58, 58, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (90, 90, 90, 255))
        
        # Input fields
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (248, 248, 248, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (255, 255, 255, 255))
        
        # Checkboxes
        dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (42, 42, 42, 255))
        
        # Headers (for collapsibles)
        dpg.add_theme_color(dpg.mvThemeCol_Header, (232, 232, 232, 255))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (221, 221, 221, 255))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (204, 204, 204, 255))

# Apply theme
dpg.bind_theme(monochrome_theme)
```
### Component Implementation
#### 1. Section Titles
```def add_section_title(text):
    """Add uppercase section title with underline"""
    dpg.add_text(text.upper(), color=(58, 58, 58, 255))
    dpg.add_separator()
    dpg.add_spacing(count=2)
```
#### 2. Form Groups (Horizontal)
```def add_horizontal_group():
    """Standard horizontal form group"""
    with dpg.group(horizontal=True):
        dpg.add_text("Label:", width=120)
        dpg.add_input_int(width=100, default_value=0)
```
#### 3. MA Type Selector (CRITICAL COMPONENT)
```def add_ma_selector(label, tag_prefix):
    """
    Creates MA type selector with checkboxes in 2 rows
    Row 1: ALL, EMA, SMA, HMA, WMA, ALMA
    Row 2: KAMA, TMA, T3, DEMA, VWMA, VWAP
    """
    dpg.add_text(label)
    dpg.add_spacing(count=1)
    
    # Container with background
    with dpg.child_window(height=80, border=True, 
                          tag=f"{tag_prefix}_container"):
        # Row 1
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="ALL", default_value=True, 
                           tag=f"{tag_prefix}_all")
            dpg.add_checkbox(label="EMA", default_value=True, 
                           tag=f"{tag_prefix}_ema")
            dpg.add_checkbox(label="SMA", default_value=True, 
                           tag=f"{tag_prefix}_sma")
            dpg.add_checkbox(label="HMA", default_value=True, 
                           tag=f"{tag_prefix}_hma")
            dpg.add_checkbox(label="WMA", default_value=True, 
                           tag=f"{tag_prefix}_wma")
            dpg.add_checkbox(label="ALMA", default_value=True, 
                           tag=f"{tag_prefix}_alma")
        
        # Row 2
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="KAMA", default_value=True, 
                           tag=f"{tag_prefix}_kama")
            dpg.add_checkbox(label="TMA", default_value=True, 
                           tag=f"{tag_prefix}_tma")
            dpg.add_checkbox(label="T3", default_value=True, 
                           tag=f"{tag_prefix}_t3")
            dpg.add_checkbox(label="DEMA", default_value=True, 
                           tag=f"{tag_prefix}_dema")
            dpg.add_checkbox(label="VWMA", default_value=True, 
                           tag=f"{tag_prefix}_vwma")
            dpg.add_checkbox(label="VWAP", default_value=True, 
                           tag=f"{tag_prefix}_vwap")
```
#### 4. Collapsible Sections
```def add_collapsible_section(title, tag):
    """Add collapsible section with dark header"""
    with dpg.collapsing_header(label=title.upper(), default_open=True, 
                               tag=tag):
        # Content goes here
        pass
```
#### 5. Parameter Groups
```def add_param_group():
    """Compact parameter group with gray background"""
    with dpg.child_window(height=40, border=True, 
                          tag="param_group_container"):
        with dpg.group(horizontal=True):
            dpg.add_text("Stop Long X:")
            dpg.add_input_int(width=70, default_value=2)
            dpg.add_text("RR:")
            dpg.add_input_int(width=70, default_value=3)
            dpg.add_text("LP:")
            dpg.add_input_int(width=70, default_value=2)
```
#### 6. Date/Time Inputs
```def add_datetime_input(label, default_date, default_time):
    """Date input with calendar button and time"""
    with dpg.group(horizontal=True):
        dpg.add_text(label, width=120)
        dpg.add_input_text(default_value=default_date, width=150)
        dpg.add_button(label="📅", width=40)  # Calendar button
        dpg.add_input_text(default_value=default_time, width=80)
```
#### 7. Buttons
```# Primary button
dpg.add_button(label="Run", width=80, height=30, 
               callback=run_backtest)

# Secondary button
dpg.add_button(label="Cancel", width=80, height=30)

# Apply secondary button theme
with dpg.theme() as secondary_btn_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (204, 204, 204, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (187, 187, 187, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (42, 42, 42, 255))
```
#### 8. Results Area
```def add_results_area():
    """Multi-line text area for results"""
    dpg.add_text("RESULTS", color=(58, 58, 58, 255))
    dpg.add_separator()
    dpg.add_spacing(count=2)
    
    with dpg.child_window(height=200, border=True, 
                          tag="results_window"):
        dpg.add_text("Нажмите 'Run' для запуска бэктеста...", 
                    color=(119, 119, 119, 255), 
                    tag="results_text")
```
### Layout Structure (Required Order)
#### 1. Date Filter Section
```with dpg.group():
    dpg.add_spacing(count=2)
    with dpg.group(horizontal=True):
        dpg.add_checkbox(label="Date Filter", default_value=True)
        dpg.add_spacing(count=3)
        dpg.add_checkbox(label="Backtester", default_value=True)
    
    dpg.add_spacing(count=2)
    add_datetime_input("Start Date", "2025-04-01", "08:00")
    add_datetime_input("End Date", "2025-09-01", "08:00")
    dpg.add_spacing(count=3)
```
#### 2. MA Settings
```with dpg.group():
    add_ma_selector("T MA Type", "t_ma")
    dpg.add_spacing(count=2)
    
    with dpg.group(horizontal=True):
        dpg.add_text("Length:", width=120)
        dpg.add_input_int(width=100, default_value=45)
    
    with dpg.group(horizontal=True):
        dpg.add_text("Close Count Long:", width=120)
        dpg.add_input_int(width=100, default_value=7)
        dpg.add_text("Close Count Short:")
        dpg.add_input_int(width=100, default_value=5)
    dpg.add_spacing(count=3)
```
#### 3. Stops and Filters (Collapsible)
```with dpg.collapsing_header(label="STOPS AND FILTERS", default_open=True):
    # Parameter groups for stops
    pass
```
#### 4. Trailing Stops (Collapsible)
```with dpg.collapsing_header(label="TRAILING STOPS", default_open=True):
    # Trail RR inputs
    # MA selectors with Length/Offset
    pass
```
#### 5. Risk Settings
```with dpg.group():
    dpg.add_spacing(count=2)
    with dpg.group(horizontal=True):
        dpg.add_text("Risk Per Trade:", width=120)
        dpg.add_input_float(width=100, default_value=2.0, step=0.01)
        dpg.add_text("Contract Size:")
        dpg.add_input_float(width=100, default_value=0.01, step=0.01)
    dpg.add_spacing(count=3)
```
#### 6. Results Area
```add_results_area()
dpg.add_spacing(count=3)
```
#### 7. Action Buttons
```with dpg.group(horizontal=True):
    dpg.add_button(label="Defaults", width=100, height=30)
    dpg.add_spacer(width=400)  # Push buttons to edges
    dpg.add_button(label="Cancel", width=80, height=30)
    dpg.add_button(label="Run", width=80, height=30)
```
### Spacing Guidelines
```# Standard spacing between sections
dpg.add_spacing(count=3)  # ~15px equivalent

# Small spacing within groups
dpg.add_spacing(count=2)  # ~10px equivalent

# Minimal spacing
dpg.add_spacing(count=1)  # ~5px equivalent
```
### DearPyGUI Style Variables
```# Apply global padding/spacing
dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 20, 20)
dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 6)
dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 10)
dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 8)
dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4)
dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 3)
```
### Input Widths
- Standard label width: 120px
- Standard input width: 100px
- Date input width: 150px
- Time input width: 80px
- Compact input (param groups): 70px
- Button width: 80-100px
- Calendar button: 40px
### Design Philosophy
1. Monochrome only - Pure grayscale palette using DearPyGUI color tuples
2. No tab navigation - Single scrollable window with all settings visible
3. Use child_window for containers - Creates bordered sections with backgrounds
4. Collapsing headers for advanced options - Keeps interface clean
5. Horizontal groups for inline layouts - Efficient space usage
6. Consistent tag naming - Use descriptive tags for all interactive elements
7. Theme-based styling - Apply monochrome theme globally
8. MA selector is checkbox grid - Critical: 2 rows, 6 columns each
### Key DearPyGUI-Specific Notes
- Use dpg.group(horizontal=True) for inline elements
- Use dpg.child_window() for bordered containers (MA selector, param groups, results)
- Use dpg.collapsing_header() for expandable sections
- All interactive elements need unique tag parameters for callbacks
- Use dpg.add_separator() after section titles
- Colors must be in (R, G, B, A) format with 0-255 values
- Widths/heights are in pixels
- Use dpg.add_spacer() for flexible spacing in horizontal groups
### Key Distinguishing Features
- ✅ Strict monochrome color scheme (RGB tuples)
- ✅ MA type selection via checkbox grid in child_window
- ✅ No tab navigation - single window layout
- ✅ Collapsing headers for advanced sections
- ✅ Child windows for parameter groups and containers
- ✅ Professional appearance with consistent spacing
- ✅ All settings accessible without switching views
- ✅ Theme-based styling for consistency