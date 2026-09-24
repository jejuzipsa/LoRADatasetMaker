from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from loradatasetmaker.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LoRA Dataset Maker")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
