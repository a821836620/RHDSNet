import argparse

from create_synthetic_dce_dataset import create_synthetic_dataset
from utils.io import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="RHDSNet unified command-line entry.")
    parser.add_argument("--command", required=True, choices=[
        "train",
        "test",
        "sota",
        "generate_then_segment",
        "ablation_db_ld",
        "ablation_mtc",
        "robustness",
        "stats",
        "visualize",
        "synthetic",
    ])
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--result_dir", default="results/five_runs")
    parser.add_argument("--dataset", default="ISPY2")
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument("--case_npz", default=None)
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--axis", default="axial", choices=["axial", "sagittal", "coronal"])
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--synthetic_cases", type=int, default=12)
    parser.add_argument("--synthetic_shape", type=int, nargs=3, default=[48, 96, 96], help="D H W")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else None

    if args.command == "train":
        from experiments.train_rhdsnet import run_train

        return run_train(cfg)
    if args.command == "test":
        from experiments.test_rhdsnet import run_test

        return run_test(cfg, ckpt=args.ckpt)
    if args.command == "sota":
        from experiments.run_sota_comparison import run_sota_comparison

        return run_sota_comparison(cfg)
    if args.command == "generate_then_segment":
        from experiments.run_generate_then_segment import run_generate_then_segment

        return run_generate_then_segment(cfg)
    if args.command == "ablation_db_ld":
        from experiments.run_ablation_db_ld import run_ablation_db_ld

        return run_ablation_db_ld(cfg)
    if args.command == "ablation_mtc":
        from experiments.run_ablation_mtc import run_ablation_mtc

        return run_ablation_mtc(cfg)
    if args.command == "robustness":
        from experiments.run_robustness_recursive_noise import run_robustness_recursive_noise

        return run_robustness_recursive_noise(cfg, ckpt=args.ckpt)
    if args.command == "stats":
        from experiments.run_statistical_test import run_statistical_test

        return run_statistical_test(args.result_dir, dataset=args.dataset, baselines=args.baselines)
    if args.command == "visualize":
        from experiments.visualize_results import visualize_from_case

        if not args.case_npz or not args.pred_dir:
            raise ValueError("--case_npz and --pred_dir are required for visualize.")
        return visualize_from_case(
            args.case_npz,
            args.pred_dir,
            args.output_dir or "outputs/visualizations",
            axis=args.axis,
            index=args.index,
        )
    if args.command == "synthetic":
        out = args.output_dir or (cfg.get("dataset", {}).get("root", "data/synthetic_ispy2") if cfg else "data/synthetic_ispy2")
        create_synthetic_dataset(out, args.synthetic_cases, cfg.get("dataset", {}).get("num_post_phases", 4) if cfg else 4, tuple(args.synthetic_shape))
        return {"output_dir": out}
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
