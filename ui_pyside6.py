import os

os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.ffmpeg=false")

from ui.app import main

if __name__ == "__main__":
    main()
