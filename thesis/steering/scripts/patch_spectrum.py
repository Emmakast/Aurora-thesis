import sys
from pathlib import Path

path = Path("thesis/scripts/plot_spectrum.py")
text = path.read_text()

# 1. Unfreeze dataclass
text = text.replace("@dataclass(frozen=True)", "@dataclass(frozen=False)")

# 2. Add ifs_mode to load_spectra_long
text = text.replace("""def load_spectra_long(
    results_dir: Path,
    cfg: AnalysisCfg,
    models: list[str] | None = None,
    exclude: set[str] | None = None,
) -> pd.DataFrame:""", """def load_spectra_long(
    results_dir: Path,
    cfg: AnalysisCfg,
    models: list[str] | None = None,
    exclude: set[str] | None = None,
    ifs_mode: bool = False,
) -> pd.DataFrame:""")

text = text.replace("""    if cfg.csv_prefix == "ke_spectrum":
        csvs = [p for p in csvs if "850hpa" not in Path(p).name]
    if not csvs:""", """    if cfg.csv_prefix == "ke_spectrum":
        csvs = [p for p in csvs if "850hpa" not in Path(p).name]
        
    if ifs_mode:
        csvs = [p for p in csvs if p.endswith("_ifs.csv")]
    else:
        csvs = [p for p in csvs if not p.endswith("_ifs.csv")]

    if not csvs:""")

# 3. Add ifs_mode to load_spectra_wide
text = text.replace("""def load_spectra_wide(
    results_dir: Path,
    cfg: AnalysisCfg,
    models: list[str] | None = None,
    exclude: set[str] | None = None,
) -> dict[str, pd.DataFrame]:""", """def load_spectra_wide(
    results_dir: Path,
    cfg: AnalysisCfg,
    models: list[str] | None = None,
    exclude: set[str] | None = None,
    ifs_mode: bool = False,
) -> dict[str, pd.DataFrame]:""")

text = text.replace("""    if cfg.csv_prefix == "ke_spectrum":
        csvs = [p for p in csvs if "850hpa" not in Path(p).name]
    if not csvs:""", """    if cfg.csv_prefix == "ke_spectrum":
        csvs = [p for p in csvs if "850hpa" not in Path(p).name]

    if ifs_mode:
        csvs = [p for p in csvs if p.endswith("_ifs.csv")]
    else:
        csvs = [p for p in csvs if not p.endswith("_ifs.csv")]

    if not csvs:""")

# 4. Add CLI arguments
text = text.replace("""    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS,
                        help="Ratio thresholds for sensitivity table (default: 0.3 0.5 0.7 0.9)")
    args = parser.parse_args()""", """    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS,
                        help="Ratio thresholds for sensitivity table (default: 0.3 0.5 0.7 0.9)")
    parser.add_argument("--ifs", action="store_true",
                        help="Use IFS HRES comparison files (*_ifs.csv) instead of ERA5")
    args = parser.parse_args()""")

# 5. Modify main logic
text = text.replace("""    cfg = ANALYSES[args.analysis]
    results_dir = Path(args.results_dir)""", """    # Create a copy so we can mutate safely if needed, or just modify since frozen=False
    cfg = ANALYSES[args.analysis]
    if args.ifs:
        cfg.title += " (vs IFS HRES)"
        cfg.ratio_ylabel = cfg.ratio_ylabel.replace("ERA5", "IFS")
    else:
        cfg.title += " (vs ERA5)"
    
    results_dir = Path(args.results_dir)""")

text = text.replace("""    df_long = load_spectra_long(results_dir, cfg, models=args.models, exclude=exclude)
    models_wide = load_spectra_wide(results_dir, cfg, models=args.models, exclude=exclude)""", """    df_long = load_spectra_long(results_dir, cfg, models=args.models, exclude=exclude, ifs_mode=args.ifs)
    models_wide = load_spectra_wide(results_dir, cfg, models=args.models, exclude=exclude, ifs_mode=args.ifs)""")

text = text.replace("""    outdir = results_dir / cfg.outdir_name""", """    outdir = results_dir / (f"{cfg.outdir_name}_IFS" if args.ifs else cfg.outdir_name)""")

# 6. Fix plot labels dynamically
text = text.replace('label="ERA5"', 'label=("IFS HRES" if "IFS" in outdir.name else "ERA5")')

path.write_text(text)
print("Patch applied.")
