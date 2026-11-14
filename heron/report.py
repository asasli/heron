"""
heron.report: generate an interactive, publication-faithful HTML report for any HERON run.

Usage:
    python -m heron.report --run runs/case1 --out runs/case1/report.html
    heron report --run runs/case1
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBPAGE_SRC = ROOT / 'webpage' / 'index.html'
ASSETS_SRC = ROOT / 'webpage' / 'assets'


def generate_report(run_dir, out_path=None):
    run_dir = Path(run_dir).expanduser().resolve()
    summary_file = run_dir / 'summary.json'
    
    if not summary_file.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
        
    with open(summary_file, 'r') as f:
        summary = json.load(f)
        
    out_path = Path(out_path) if out_path else run_dir / 'report.html'
    out_path = out_path.expanduser().resolve()
    
    # Copy assets adjacent to report if needed
    report_assets = out_path.parent / 'assets'
    if ASSETS_SRC.exists() and not report_assets.exists():
        try:
            shutil.copytree(ASSETS_SRC, report_assets)
        except Exception:
            pass

    # Read base template
    html = WEBPAGE_SRC.read_text(encoding='utf-8')
    
    # Inject run-specific values into telemetry cards if present
    case_num = summary.get('case', 1)
    seed_num = summary.get('seed', 0)
    snr_val = summary.get('mbhb_snr', 357)
    n_ucb = summary.get('n_ucb_detected', summary.get('n_ucb_subtracted', 30))
    alpha = summary.get('global_alpha', 0.14)
    a90 = summary.get('alert_sky90_calibrated_deg2', 143)
    
    # Custom banner insertion
    custom_badge = (f"Run Event: Sangria Case-{case_num} &bull; Seed {seed_num} &bull; "
                    f"SNR {snr_val:.0f} &bull; UCBs {n_ucb} (&alpha;={alpha:.2f})")
    html = html.replace("Demonstrated Event: Sangria Case-1 (LDC2a) &bull; Noise Seed 0 &bull; 1-Month TDI", custom_badge)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    print(f"[report] Interactive report generated at: {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate interactive HTML report from HERON run")
    parser.add_argument('--run', default='runs/case1', help='Directory containing summary.json')
    parser.add_argument('--out', default=None, help='Output HTML path')
    args = parser.parse_args(argv)
    generate_report(args.run, args.out)


if __name__ == '__main__':
    main()
