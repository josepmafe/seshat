"""Render architecture-diagram.html to PNG via Playwright (headless Chromium)."""

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
HTML = HERE / "architecture-diagram.html"
PNG = HERE / "architecture-diagram.png"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=3)
        page.goto(HTML.as_uri())
        svg = page.locator("svg")
        svg.screenshot(path=str(PNG))
        browser.close()

    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
