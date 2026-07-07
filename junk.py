import time
import threading
import itertools
import random


SPINNERS = {
    # Clean terminal spinners
    "classic": ["|", "/", "-", "\\"],
    "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "dots2": ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
    "circle": ["◐", "◓", "◑", "◒"],
    "circle2": ["◴", "◷", "◶", "◵"],
    "arrows": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    "triangles": ["◢", "◣", "◤", "◥"],
    "blocks": ["▖", "▘", "▝", "▗"],
    "bar": ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"],
    "bounce": ["⠁", "⠂", "⠄", "⠂"],

    # Text effects
    "pulse": ["·  ", "·· ", "···", " ··", "  ·", "   "],
    "ellipsis": [".  ", ".. ", "...", " ..", "  .", "   "],
    "loading": ["loading   ", "loading.  ", "loading.. ", "loading..."],

    # Flashier Unicode
    "moon": ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],
    "earth": ["🌍", "🌎", "🌏"],
    "clock": ["🕛", "🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚"],
    "weather": ["☀️", "🌤️", "⛅", "🌥️", "☁️", "🌧️", "⛈️", "🌧️", "☁️", "🌥️", "⛅", "🌤️"],
    "hearts": ["♡", "♥"],
    "sparkle": ["✦", "✧", "✨", "✧"],
    "stars": ["☆", "✩", "✪", "✩"],
    "fire": ["🔥", "🟠", "🟡", "⚪", "🟡", "🟠"],
    "dna": ["A", "T", "C", "G"],

    # Fun icons
    "rocket": ["🚀    ", " 🚀   ", "  🚀  ", "   🚀 ", "    🚀", "   🚀 ", "  🚀  ", " 🚀   "],
    "snake": ["🐍    ", " 🐍   ", "  🐍  ", "   🐍 ", "    🐍", "   🐍 ", "  🐍  ", " 🐍   "],
    "brain": ["🧠", "💭", "🧠", "💡"],
    "microscope": ["🔬", "🧬", "🧫", "🧪"],
}


def format_elapsed(seconds):
    seconds = int(seconds)
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)

    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    return f"{mins:02d}:{secs:02d}"


def choose_spinner_style(style="random", seed=None):
    styles = list(SPINNERS.keys())

    if style is None or style == "random":
        rng = random.Random(seed)
        return rng.choice(styles)

    if style == "random_safe":
        # Avoid emoji styles that may render weirdly in some terminals.
        safe_styles = [
            "classic",
            "dots",
            "dots2",
            "circle",
            "circle2",
            "arrows",
            "triangles",
            "blocks",
            "bar",
            "pulse",
            "ellipsis",
            "loading",
        ]
        rng = random.Random(seed)
        return rng.choice(safe_styles)

    if style not in SPINNERS:
        raise ValueError(
            f"Unknown spinner style: {style!r}. "
            f"Available styles: {sorted(SPINNERS.keys())}"
        )

    return style


def spinner(message, stop_event, start_time, interval=0.15, style="dots"):
    frames = SPINNERS[style]
    wheel = itertools.cycle(frames)

    while not stop_event.is_set():
        elapsed = format_elapsed(time.time() - start_time)
        frame = next(wheel)

        print(
            f"\r{frame} {message} | style: {style} | elapsed: {elapsed}",
            end="",
            flush=True,
        )

        time.sleep(interval)

    elapsed = format_elapsed(time.time() - start_time)
    print(f"\r✓ {message} done | style: {style} | elapsed: {elapsed}    ")


def run_with_spinner(
    message,
    func,
    *args,
    interval=0.15,
    style="random",
    seed=None,
    **kwargs,
):
    chosen_style = choose_spinner_style(style=style, seed=seed)

    stop_event = threading.Event()
    start_time = time.time()

    thread = threading.Thread(
        target=spinner,
        args=(message, stop_event, start_time, interval, chosen_style),
        daemon=True,
    )

    thread.start()

    try:
        result = func(*args, **kwargs)
    finally:
        stop_event.set()
        thread.join()

    return result