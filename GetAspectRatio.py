from screeninfo import get_monitors

def detect_monitor():
    monitors = get_monitors()
    print("Available Displays:")
    for i, m in enumerate(monitors):
        print(f"[{i}] {m.width}x{m.height} {'(Primary)' if m.is_primary else ''}")

    choice = 0
    if len(monitors) > 1:
        choice = int(input("Select display index: "))

    selected_monitor = monitors[choice]
    print(f"Selected display: {selected_monitor.width}x{selected_monitor.height}")
    return selected_monitor.width, selected_monitor.height

if __name__ == "__main__":
    width, height = detect_monitor()
    print(f"Chosen resolution: {width}x{height}")