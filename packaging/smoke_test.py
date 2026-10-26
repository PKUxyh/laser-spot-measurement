"""Exercise the frozen GUI, file readers, analysis and plotting without dialogs."""
import json
from pathlib import Path
import tempfile
import time
import traceback


def run(report_path):
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    root = None
    report = {'passed': False}
    try:
        import tkinter as tk
        import numpy as np
        import pandas as pd
        import tifffile
        import app_gui

        root = tk.Tk()
        root.withdraw()
        app = app_gui.FocusAnalysisApp(root)
        errors = []
        root.report_callback_exception = lambda *exc: errors.append(
            ''.join(traceback.format_exception(*exc)))
        app_gui.messagebox.showerror = lambda title, message, **kw: errors.append(str(message))
        results = []
        finish = app._finish_analysis

        def record_result(result=None, error=None):
            finish(result=result, error=error)
            if result is not None:
                results.append(result)

        app._finish_analysis = record_result
        figures = []
        add_plot = app_gui.add_plot

        def record_plot(*args, **kwargs):
            panel = add_plot(*args, **kwargs)
            figures.append(args[2])
            return panel

        app_gui.add_plot = record_plot

        def wait_for_analysis(mode):
            deadline = time.monotonic() + 90
            while app.busy and not errors and time.monotonic() < deadline:
                root.update()
                time.sleep(0.01)
            if errors:
                raise RuntimeError('\n'.join(errors))
            assert not app.busy, 'Analysis timed out'
            assert app.current_mode == mode
            assert len(figures) == 4
            assert len(app.display_frame.winfo_children()) == 4
            for figure in figures:
                assert figure.canvas.toolbar is not None
                figure.canvas.toolbar.home()
                # A withdrawn Tk window can report a one-pixel holder size.
                # Set an explicit export size for the non-interactive check.
                figure.set_size_inches(app_gui.SINGLE_COLUMN_SIZE, forward=False)
                figure.canvas.draw()
                assert min(figure.canvas.get_width_height()) > 200
            figures[-1].savefig(report_path.parent / f'smoke-{mode}.png')

        with tempfile.TemporaryDirectory(prefix='focus smoke ') as folder:
            y, x = np.indices((100, 120))
            paths = []
            for shot, center_x in enumerate((58, 60, 62), start=1):
                data = np.rint(4000 * np.exp(-0.5 * (
                    ((x - center_x) / 14) ** 2 + ((y - 50) / 9) ** 2))).astype(np.uint16)
                path = Path(folder) / f'光斑 shot {shot}.tif'
                if shot == 3:
                    path = path.with_suffix('.csv')
                    pd.DataFrame(data).to_csv(path, index=False)
                else:
                    tifffile.imwrite(path, data)
                paths.append(path)

            app.analyze_file(paths[0])
            wait_for_analysis('single')
            assert np.isfinite(results[-1]['a0']) and results[-1]['a0'] > 0
            figures.clear()
            app.analyze_batch(paths)
            wait_for_analysis('batch')
            batch = results[-1]
            assert len(batch['file_paths']) == 3 and not batch['errors']
            assert batch['stats']['jitter_rms'] > 0
            assert batch['mean_data'].shape == (100, 120)
            report.update(passed=True, single_panels=4, batch_panels=4,
                          analyzed_shots=3, formats=['TIFF', 'CSV'],
                          unicode_paths=True, navigation_toolbars=True,
                          batch_stats=batch['stats'])
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        if root is not None:
            root.destroy()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report['passed'] else 1
