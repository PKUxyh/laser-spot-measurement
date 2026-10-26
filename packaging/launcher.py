"""Portable application entry point and release verification switch."""
import sys


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
        from smoke_test import run
        raise SystemExit(run(sys.argv[2]))
    from app_gui import main
    main()
