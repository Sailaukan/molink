import argparse

from molink.sampler import MolinkSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "de_novo",
            "scaffold_decoration",
            "linker_design",
            "motif_extension",
            "scaffold_morphing",
        ],
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)

    parser.add_argument("--scaffold")
    parser.add_argument("--motif")
    parser.add_argument("--fragment-a")
    parser.add_argument("--fragment-b")
    parser.add_argument("--side-chains")
    parser.add_argument("--mol")
    parser.add_argument("--core")
    args = parser.parse_args()

    sampler = MolinkSampler(args.checkpoint)

    if args.task == "de_novo":
        results = sampler.de_novo_generation(
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.task == "scaffold_decoration":
        if not args.scaffold:
            raise ValueError("--scaffold is required for scaffold_decoration")
        results = sampler.scaffold_decoration(
            args.scaffold,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.task == "motif_extension":
        if not args.motif:
            raise ValueError("--motif is required for motif_extension")
        results = sampler.motif_extension(
            args.motif,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.task == "linker_design":
        if not args.fragment_a or not args.fragment_b:
            raise ValueError("--fragment-a and --fragment-b are required for linker_design")
        results = sampler.linker_design(
            args.fragment_a,
            args.fragment_b,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.task == "scaffold_morphing":
        side_chains = None
        if args.side_chains:
            side_chains = [s.strip() for s in args.side_chains.split(",") if s.strip()]
        results = sampler.scaffold_morphing(
            side_chains=side_chains,
            mol=args.mol,
            core=args.core,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    for smi in results:
        print(smi)


if __name__ == "__main__":
    main()
